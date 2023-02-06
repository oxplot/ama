package main

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"

	_ "github.com/oxplot/starenv/autoload"
	gogpt "github.com/sashabaranov/go-gpt3"
)

var (
	openAIAuthToken = os.Getenv("OPENAI_API_KEY")
	clickupAPIToken = os.Getenv("CLICKUP_API_KEY")

	indexFolderFlag = flag.Bool("index", false, "Index file paths passed on stdin")
	cliQuery        = flag.String("query", "", "Run in command line mode")
)

// Vector is a vector of floats.
type Vector []float64

// Distance returns the distance between two vectors.
func (v Vector) Distance(v2 Vector) float64 {
	var sum float64
	for i, x := range v {
		sum += (x - v2[i]) * (x - v2[i])
	}
	return sum
}

// Document is a document to be indexed.
type Document struct {
	Title string `json:"title"`
	Link  string `json:"link"`
	HTML  string `json:"html"`
}

var (
	htmlTagPat     = regexp.MustCompile(`<[^>]*>`)
	htmlHeadersPat = regexp.MustCompile(`<[hH]\d+[^>]*>`)
	htmlScriptPat  = regexp.MustCompile(`<script[^>]*>.*?</script>`)
)

// Chunks returns the document content split into chunks.
func (d *Document) Chunks() []string {
	v := strings.ReplaceAll(d.HTML, "\n", " ")
	v = htmlScriptPat.ReplaceAllString(v, " ")
	chunks := htmlHeadersPat.Split(v, -1)
	for i, c := range chunks {
		chunks[i] = htmlTagPat.ReplaceAllString(c, " ")
	}
	return chunks
}

// ChunkRef is a reference to a document chunk.
type ChunkRef struct {
	DocumentID  int `json:"doc"`
	ChunkNumber int `json:"chunk"`
}

// Embedding is a vector embedding of a document chunk.
type Embedding struct {
	Vector  Vector `json:"vec"`
	ChunkID int    `json:"chunk"`
}

// Index is an index of documents.
type Index struct {
	Documents  []string    `json:"docs"`
	Embeddings []Embedding `json:"embeddings"`
	// ChunkHash -> ChunkRef
	ChunkRefs []ChunkRef `json:"chunks"`
}

// runIndex indexes a folder of documents recursively.
func runIndex() error {

	ctx := context.Background()

	if openAIAuthToken == "" {
		return fmt.Errorf("OPENAI_API_KEY is not set")
	}
	aiCl := gogpt.NewClient(openAIAuthToken)

	docs := make([]string, 0)
	embeddings := make([]Embedding, 0)
	chunkRefs := make([]ChunkRef, 0)

	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		path := sc.Text()

		f, err := os.Open(path)
		if err != nil {
			return err
		}

		doc := Document{}
		if err := json.NewDecoder(f).Decode(&doc); err != nil {
			f.Close()
			return err
		}
		f.Close()

		// For each chunk, get the embedding.

		chunks := doc.Chunks()
		resp, err := aiCl.CreateEmbeddings(ctx, gogpt.EmbeddingRequest{
			Input: chunks,
			Model: gogpt.AdaEmbeddingV2,
		})
		if err != nil {
			return err
		}

		for i, emb := range resp.Data {
			embeddings = append(embeddings, Embedding{
				Vector:  emb.Embedding,
				ChunkID: len(chunkRefs),
			})
			chunkRefs = append(chunkRefs, ChunkRef{
				DocumentID:  len(docs),
				ChunkNumber: i,
			})
		}

		docs = append(docs, path)

		log.Printf("- indexed %s (%d chunks)", path, len(chunks))
	}
	if sc.Err() != nil {
		log.Printf("warning: failed to index all documents: %v", sc.Err())
	}

	// Construct and store the index

	idx := Index{
		Documents:  docs,
		Embeddings: embeddings,
		ChunkRefs:  chunkRefs,
	}
	f, err := os.Create("index.json.gz")
	if err != nil {
		return err
	}
	gz := gzip.NewWriter(f)
	defer gz.Close()
	if err := json.NewEncoder(gz).Encode(idx); err != nil {
		return err
	}
	return gz.Flush()
}

func runServer() error {

	if openAIAuthToken == "" {
		return fmt.Errorf("OPENAI_API_KEY is not set")
	}
	aiCl := gogpt.NewClient(openAIAuthToken)

	_ = aiCl

	return nil
}

func loadIndex() (idx Index, err error) {
	f, err := os.Open("index.json.gz")
	if err != nil {
		return
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return
	}
	err = json.NewDecoder(gz).Decode(&idx)
	return
}

func runCLI(query string) error {

	// Load the index

	idx, err := loadIndex()
	if err != nil {
		return fmt.Errorf("failed to load index: %w", err)
	}

	// Get the embedding for the query

	ctx := context.Background()
	aiCl := gogpt.NewClient(openAIAuthToken)
	resp, err := aiCl.CreateEmbeddings(ctx, gogpt.EmbeddingRequest{
		Input: []string{query},
		Model: gogpt.AdaEmbeddingV2,
	})
	if err != nil {
		return fmt.Errorf("failed to get embedding for query: %w", err)
	}
	qEmb := resp.Data[0].Embedding

	// Sort the documents by closest distance to the query

	sort.Slice(idx.Embeddings, func(i, j int) bool {
		return idx.Embeddings[i].Vector.Distance(qEmb) < idx.Embeddings[j].Vector.Distance(qEmb)
	})

	// Join the chunk content of the first 3 chunks.

	var chunks []string
	for i := 0; i < 3; i++ {
		chunkRef := idx.ChunkRefs[idx.Embeddings[i].ChunkID]
		docPath := idx.Documents[chunkRef.DocumentID]
		f, err := os.Open(docPath)
		if err != nil {
			return err
		}
		var doc Document
		if err := json.NewDecoder(f).Decode(&doc); err != nil {
			f.Close()
			return err
		}
		f.Close()
		chunks = append(chunks, doc.Chunks()[chunkRef.ChunkNumber])
	}

	// Create the prompt

	prompt := `Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"\n\n`
	prompt += fmt.Sprintf("Context: %s\n\n", strings.Join(chunks, " "))
	prompt += fmt.Sprintf("Q: %s\nA:", query)

	// Generate the answer

	compResp, err := aiCl.CreateCompletion(ctx, gogpt.CompletionRequest{
		Prompt:           prompt,
		Model:            gogpt.GPT3TextDavinci003,
		MaxTokens:        300,
		TopP:             1,
		FrequencyPenalty: 0,
		Temperature:      0,
		PresencePenalty:  0,
	})
	if err != nil {
		return fmt.Errorf("completion failed: %w", err)
	}

	fmt.Printf("Answer: %s\n", compResp.Choices[0].Text)
	return nil
}

func main() {
	log.SetFlags(0)
	flag.Parse()

	if *indexFolderFlag {
		if err := runIndex(); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *cliQuery != "" {
		if err := runCLI(*cliQuery); err != nil {
			log.Fatal(err)
		}
	}

	if err := runServer(); err != nil {
		log.Fatal(err)
	}
}
