# Individual Assignment: Medium Article RAG Assistant

## Goal

Create a knowledgeable AI assistant specialised in **Medium articles** using a **Retrieval-Augmented Generation (RAG)** system.

Your assistant must answer a set of questions **only from the provided article dataset**, without relying on any external information.

---

## Dataset

You will work with a dataset of English-language articles published on Medium.

The file provided to you is a single CSV containing roughly **7,600 English articles**, drawn from the larger public Medium Articles dataset.

### Schema (CSV columns)

```text
title, text, url, authors, timestamp, tags
```

Dataset file:

```text
medium-english-50mb.csv
```

---

# Functional Requirements: Query Capabilities

Your RAG system should accurately answer several distinct categories of questions using only the dataset (metadata + article passages).

The examples below are grounded in the kinds of articles present in the corpus.

---

## 1. Precise Fact Retrieval

### Goal

The model must locate a single, specific article based on semantic criteria within the corpus.

### Example

> Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author.

### Explanation

The assistant should locate one concrete article and return the requested fields.

---

## 2. Multi-Result Topic Listing (up to 3 results)

### Goal

Return multiple distinct article titles that match a theme or topic.

### Example

> List exactly 3 articles about education. Return only the titles.

### Explanation

The assistant must retrieve multiple distinct articles, not multiple chunks of the same article.

No need to support lists larger than 3.

---

## 3. Key Idea Summary Extraction

### Goal

Identify a relevant article and generate a concise summary of its main idea.

### Example

> Find an article that argues past pandemics (such as the bubonic plague) can spur innovation and recovery, and summarise its central argument.

### Explanation

The assistant should provide the key idea based on retrieved text chunks (not necessarily the whole article).

---

## 4. Recommendation with Evidence-Based Justification

### Goal

Recommend one relevant article and justify the choice.

### Example

> I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?

### Explanation

The assistant should choose an article and provide a justification grounded in the retrieved text.

---

## Important Constraint

Your system must answer these questions **without relying on the model's own background knowledge**.

---

# Tools, Budget & Constraints

## Available Models

### Embedding Model

```text
4UHRUIN-text-embedding-3-small
```

Default dimensions:

```text
1536
```

### Chat Model

```text
4UHRUIN-gpt-5-mini
```

---

## Budget Constraint

Your total budget for this assignment is:

```text
5 USD
```

This budget includes **all development and testing**.

Recommendations:

- Avoid embedding the same data repeatedly.
- Start with a smaller subset.
- Validate your approach.
- Scale up only after verification.

**Overstepping the budget may reduce your final score.**

---

# RAG Hyperparameters

You must choose and report the following:

| Parameter | Maximum |
|------------|-----------|
| Chunk Size | 1024 tokens |
| Overlap | 30% (0.3) |
| Top-k | 30 |

Pushing unnecessary data into the model context will be considered inefficient.

---

# Required System Prompt for GPT-5 Mini

You must include the following system prompt section when calling `4UHRUIN-gpt-5-mini`.

You may add clarifications (e.g., response style), but the constraints below must remain intact.

```text
You are a Medium-article assistant that answers questions strictly and only
based on the Medium articles dataset context provided to you (metadata
and article passages). You must not use any external knowledge, the open
internet, or information that is not explicitly contained in the retrieved
context.

If the answer cannot be determined from the provided context,
respond:

"I don’t know based on the provided Medium articles data."

Always explain your answer using the given context, quoting or
paraphrasing the relevant article passage or metadata when helpful.
```

---

# Vector Database & Deployment

## Pinecone

Use Pinecone as your vector database:

- https://www.pinecone.io
- Make sure your Pinecone index remains active until you receive a grade or are otherwise instructed.
- Make sure to set dimensions according to the embedding model.

---

## Vercel

Deploy your application to Vercel:

- https://vercel.com
- A **public live URL** must be submitted.

---

# API Requirements

## 1. POST `{your-url}/api/prompt`

Used to query your system with natural-language questions.

### Input

```json
{
  "question": "Your natural language question here"
}
```

### Output

```json
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Sample article title",
      "chunk": "article chunk retrieved",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "the system prompt used to query the chat model",
    "User": "the user prompt used to query the chat model"
  }
}
```

### Field Descriptions

#### response

What GPT-5 Mini returns after using the retrieved context.

#### context

Array of retrieved context chunks.

#### Augmented_prompt

The final prompt sent to GPT-5 Mini.

Contains:

- System prompt
- User prompt

---

## 2. GET `{your-url}/api/stats`

Returns the configuration currently used by your RAG system.

### Required JSON Format

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 7
}
```

### Field Descriptions

| Field | Description |
|---------|-------------|
| chunk_size | Integer (tokens or approximate tokens/chars as defined in code) |
| overlap_ratio | Number between 0 and 0.3 |
| top_k | Integer between 1 and 30 |

If you later change your hyperparameters, this endpoint must always reflect the current values.

---

# Deliverable & Deadline

Submit:

1. Public live URL
2. Public GitHub URL

Deadline:

```text
7.6.2026 – End of Day
```

---

# Notes

- Start with a small subset of articles.
- Verify your RAG pipeline end-to-end before scaling.
- Experiment explicitly with chunk size and overlap.
- Compare several settings.
- Report which settings retrieve the most relevant passages for the required question types.
- Avoid re-embedding the entire corpus whenever you tweak parameters.
- Design a workflow that minimizes embedding costs while still allowing experimentation.

---

Good luck, and have fun building your Medium Article RAG Assistant!
