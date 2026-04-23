# Lectura: Retrievers interesantes en LangChain
Los retrievers constituyen el corazón de cualquier sistema RAG (Retrieval-Augmented Generation) exitoso. LangChain ha evolucionado para ofrecer una amplia gama de retrievers sofisticados que van mucho más allá de la simple búsqueda por similitud. En este artículo, exploraremos los retrievers más interesantes y potentes que están transformando la manera en que nuestras aplicaciones de IA acceden y procesan información.

### ¿Qué es un Retriever en LangChain?
Un retriever en LangChain es una interfaz que recibe una consulta en lenguaje natural y devuelve documentos relevantes. A diferencia de los vector stores, los retrievers no necesitan almacenar documentos, solo recuperarlos. Esta abstracción permite crear sistemas de recuperación flexibles y modulares que se adaptan a diferentes necesidades.

### Características clave:
* Entrada: string de consulta
* Salida: lista de objetos Document
* Implementan la interfaz Runnable estándar
* Compatibles con LCEL (LangChain Expression Language)

---
# 1. MultiQueryRetriever: La Perspectiva Múltiple

**¿Qué hace?**

El MultiQueryRetriever aborda las limitaciones de la búsqueda por similitud basada en distancia generando múltiples "perspectivas" alternativas de tu consulta original. En lugar de hacer una sola búsqueda, utiliza un LLM para crear varias versiones de la misma pregunta y luego combina los resultados.

>**Implementación:** 1.MultiQueryRetriever.py

### Cuándo usarlo
* Cuando tu consulta original puede ser interpretada de múltiples formas
* Para mejorar la diversidad de resultados recuperados
* En casos donde la consulta inicial podría no capturar todos los aspectos relevantes

---
# 2. ContextualCompressionRetriever: El Filtro Inteligente
### ¿Qué problema resuelve?
Uno de los desafíos con la recuperación es que normalmente no conoces las consultas específicas que enfrentará tu sistema de almacenamiento de documentos cuando ingieres datos al sistema. Esto significa que la información más relevante para una consulta puede estar enterrada en un documento con mucho texto irrelevante.

>**Implementación:** 2.ContextualCompressionRetriever.py

>**Pipeline avanzado de compresión:** 2.1ContextualCompressionRetrieverAdvanced.py

### Beneficios
* Reduce costos de llamadas a LLM al eliminar texto irrelevante
* Mejora la calidad de las respuestas al proporcionar contexto más preciso
* Permite pasar más documentos relevantes dentro del límite de tokens

---
# 3. EnsembleRetriever: El Mejor de Dos Mundos
### La potencia del híbrido

El EnsembleRetriever soporta la combinación de resultados de múltiples retrievers. Los EnsembleRetrievers reordenan los resultados de los retrievers constituyentes basándose en el algoritmo de Fusión de Ranking Recíproco.

> **Implementación BM25 + Vector Search:** 3.BM25 + Vector Search.py

### Por qué es efectivo
* BM25: Excelente para coincidencias exactas de palabras clave
* Vector Search: Superior para similitud semántica
* Fusión: Combina las fortalezas de ambos enfoques

### Casos de uso ideales
* Búsquedas que requieren tanto precisión léxica como semántica
* Documentos técnicos con terminología específica
* Sistemas de búsqueda empresarial

# 4. ParentDocumentRetriever: Precisión con Contexto
### El equilibrio perfecto

El ParentDocumentRetriever logra ese equilibrio dividiendo y almacenando pequeños trozos de datos. Durante la recuperación, primero obtiene los trozos pequeños pero luego busca los IDs padre de esos trozos y devuelve esos documentos más grandes.

> **Implementación:** 4.ParentDocumentRetriever.py

### Ventajas clave
* Embeddings precisos: Los chunks pequeños crean embeddings más representativos
* Contexto completo: Devuelve documentos padre con contexto suficiente
* Flexibilidad: Puedes ajustar el tamaño de chunks padre e hijo independientemente

# 5. SelfQueryRetriever: Búsqueda Estructurada Inteligente
### Más allá de la similitud semántica

SelfQueryRetriever utilizará un LLM para generar una consulta que es potencialmente estructurada, por ejemplo, puede construir filtros para la recuperación además de la selección habitual dirigida por similitud semántica.

>**Implementación** SelfQueryRetriever.py

**Casos de uso**
* Bases de datos con metadatos ricos
* Consultas que combinan contenido y filtros
* Sistemas que requieren búsqueda estructurada automática

# 6. TimeWeightedVectorStoreRetriever: Memoria que Desvanece

### Para información sensible al tiempo
Este retriever asigna mayor importancia a documentos más recientes, simulando cómo funciona la memoria humana.

> **implementaciión:** TimeWeightedVectorStoreRetriever.py

# 7. Técnicas Avanzadas y Combinaciones
### Retrieval con Reranking
>**Implementación:** 7.Técnicas Avanzadas y Combinaciones.py
>**Implementación:** 7.1Retrieval MMR.py