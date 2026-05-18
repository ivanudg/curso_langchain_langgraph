from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Definir el estado
class State(TypedDict):
    numero: int
    resultado: str

graph = StateGraph(State)

# Definir los nodos del workflow
def caso_par( state: State ) -> State:
    return { 'resultado': 'El número es par' }

def caso_impar( state: State ) -> State:
    return { 'resultado': 'El número es impar' }

graph.add_node( 'Par', caso_par )
graph.add_node( 'Impar', caso_impar )

# Definira la función de routing para decidir la rama de ejecución
def decidir_rama( state: State ):
    if state['numero'] % 2 == 0:
        return 'Par'
    else:
        return 'Impar'
    
# Añadir el edge condicional al workflow
graph.add_conditional_edges( START, decidir_rama )

# Conectar ambos casos al final
graph.add_edge( "Par", END )
graph.add_edge( "Impar", END )

compiled = graph.compile()

# Probar el grafo con ejemplos
print( compiled.invoke({"numero": 7})["resultado"] )