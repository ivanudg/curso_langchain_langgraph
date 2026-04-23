# Balancear relevancia con diversidad
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "lambda_mult": 0.7  # Balance entre relevancia (1.0) y diversidad (0.0)
    }
)