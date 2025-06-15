import json
import re

def extract_notebook_structure(notebook_path):
    """Extrae la estructura del notebook y contenido principal"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    structure = {
        'title': '',
        'sections': [],
        'code_cells': 0,
        'markdown_cells': 0,
        'total_cells': len(notebook['cells'])
    }
    
    current_section = None
    section_counter = 0
    
    for i, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']
        
        if cell_type == 'code':
            structure['code_cells'] += 1
        elif cell_type == 'markdown':
            structure['markdown_cells'] += 1
            
        # Buscar headers en markdown
        if cell_type == 'markdown' and cell['source']:
            content = ''.join(cell['source'])
            
            # Buscar títulos principales (# ## ###)
            if content.strip().startswith('#'):
                lines = content.split('\n')
                for line in lines:
                    if line.strip().startswith('#'):
                        level = len(line) - len(line.lstrip('#'))
                        title = line.strip('#').strip()
                        
                        if level <= 2:  # Solo nivel 1 y 2
                            section_info = {
                                'level': level,
                                'title': title,
                                'cell_index': i,
                                'content_preview': content[:200] + '...' if len(content) > 200 else content
                            }
                            
                            if level == 1:
                                structure['sections'].append(section_info)
                                current_section = section_info
                                current_section['subsections'] = []
                            elif level == 2 and current_section:
                                current_section['subsections'].append(section_info)
                        
                        break  # Solo el primer header por celda
    
    return structure

# Extraer estructura
notebook_path = "Z:/Iris/01-Learning-Datos/Big-Data/0_Repositorios-Git-Hub/Deep-Learning/deep_learning/Practica_DL_solución.ipynb"
structure = extract_notebook_structure(notebook_path)

print("ESTRUCTURA DEL NOTEBOOK")
print("=" * 50)
print(f"Total de celdas: {structure['total_cells']}")
print(f"Celdas de código: {structure['code_cells']}")
print(f"Celdas de markdown: {structure['markdown_cells']}")
print()

print("ÍNDICE DE CONTENIDOS:")
print("-" * 30)

for i, section in enumerate(structure['sections'], 1):
    print(f"{i}. {section['title']}")
    for j, subsection in enumerate(section.get('subsections', []), 1):
        print(f"   {i}.{j}. {subsection['title']}")
    print()
