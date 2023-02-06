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
	"strings"

	_ "github.com/oxplot/starenv/autoload"
	gogpt "github.com/sashabaranov/go-gpt3"
)

var (
	openAIAuthToken = os.Getenv("OPENAI_API_KEY")
	clickupAPIToken = os.Getenv("CLICKUP_API_KEY")

	indexFolderFlag = flag.Bool("index", false, "Index file paths passed on stdin")
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
		defer f.Close()

		doc := Document{}
		if err := json.NewDecoder(f).Decode(&doc); err != nil {
			return err
		}

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
	defer f.Close()
	gz := gzip.NewWriter(f)
	if err := json.NewEncoder(gz).Encode(idx); err != nil {
		return err
	}
	return f.Sync()
}

func runServer() error {

	if openAIAuthToken == "" {
		return fmt.Errorf("OPENAI_API_KEY is not set")
	}
	aiCl := gogpt.NewClient(openAIAuthToken)

	_ = aiCl

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

	if err := runServer(); err != nil {
		log.Fatal(err)
	}
}
