# Adaptive RAG 101

Welcome to Adaptive RAG 101! 

## Introduction
Welcome to Adaptive RAG 101! In this session, we'll walk through a fun example setting up an Adaptive RAG agent in LangGraph.

## Context

At LangChain, we aim to make it easy to build LLM applications. One type of LLM application you can build is an agent. There’s a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible. 

In practice though, it is incredibly difficult to build systems that reliably execute on these tasks. As we’ve worked with our users to put agents into production, we’ve learned that more control is often necessary. You might need an agent to always call a specific tool first or use different prompts based on its state.

To tackle this problem, we’ve built [LangGraph](https://langchain-ai.github.io/langgraph/) — a framework for building agent and multi-agent applications. Separate from the LangChain package, LangGraph’s core design philosophy is to help developers add better precision and control into agent workflows, suitable for the complexity of real-world systems.

## Pre-work

### Clone the Adaptive RAG 101 repo
```
git clone https://github.com/langchain-ai/adaptive-rag-101.git
```

Follow the setup.ipynb notebook and follow instructions there! If you run into issues with setting up the python environment or acquiring the necessary API keys due to any restrictions (ex. corporate policy), contact your LangChain representative and we'll find a work-around!

