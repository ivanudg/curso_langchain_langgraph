from pathlib import Path
from langchain_community.document_loaders import CSVLoader
 
csv = Path( Path.cwd(), 'Tema3', 'insumos', 'datos_ventas_small.csv' )
# Configuración avanzada para CSV
loader = CSVLoader(
    file_path=csv,
    csv_args={
        'delimiter': ';',
        'quotechar': '"',
        'fieldnames': ['ID', 'Cantidad', 'Precio unitario', 'Venta total', 'Fecha compra', 'Estado', 'Línea Producto', 'Código Producto', 'Nombre cliente', 'Ciudad', 'País', 'Territorio', 'Tamaño pedido']
    },
    encoding='utf-8',
    source_column='ID',  # Usar una columna como identificador
    metadata_columns=['Cantidad', 'Precio unitario', 'Fecha compra']  # Incluir en metadatos
)
 
docs = loader.load()
print(f"Registros de ventas cargados: {len(docs)}")
 
# Análisis de los datos cargados
productos = set()
clientes = set()
ventas_por_fecha = {}
 
for doc in docs[:10]:  # Mostrar primeros 10 registros
    print(f"\nRegistro: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
    
    # Recopilar estadísticas
    if 'Fecha compra' in doc.metadata:
        fecha = doc.metadata['Fecha compra']
        ventas_por_fecha[fecha] = ventas_por_fecha.get(fecha, 0) + 1
    
    if 'cliente' in doc.metadata:
        clientes.add(doc.metadata['cliente'])
 
print(f"\nResumen de datos:")
print(f"  Clientes únicos: {len(clientes)}")
print(f"  Fechas con ventas: {len(ventas_por_fecha)}")
print(f"  Promedio de ventas por día: {len(docs) / len(ventas_por_fecha):.1f}")