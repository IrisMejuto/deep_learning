import json
import re
from pathlib import Path

def analyze_notebook_comprehensive(notebook_path):
    """Análisis comprehensivo del notebook"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    analysis = {
        'metadata': notebook.get('metadata', {}),
        'cells': [],
        'imports': [],
        'functions': [],
        'classes': [],
        'variables': [],
        'visualizations': [],
        'models': [],
        'sections': []
    }
    
    current_section = ""
    
    for i, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']
        source = ''.join(cell['source']) if cell['source'] else ""
        
        cell_info = {
            'index': i,
            'type': cell_type,
            'source_length': len(source),
            'source_preview': source[:500] + "..." if len(source) > 500 else source
        }
        
        # Analizar contenido específico
        if cell_type == 'code' and source.strip():
            # Buscar imports
            import_lines = re.findall(r'^(?:from .+ )?import .+$', source, re.MULTILINE)
            analysis['imports'].extend(import_lines)
            
            # Buscar definiciones de funciones
            function_defs = re.findall(r'^def\s+(\w+)\s*\([^)]*\):', source, re.MULTILINE)
            analysis['functions'].extend([(func, i) for func in function_defs])
            
            # Buscar definiciones de clases
            class_defs = re.findall(r'^class\s+(\w+)(?:\([^)]*\))?:', source, re.MULTILINE)
            analysis['classes'].extend([(cls, i) for cls in class_defs])
            
            # Buscar variables importantes
            var_assignments = re.findall(r'^(\w+)\s*=', source, re.MULTILINE)
            analysis['variables'].extend([(var, i) for var in var_assignments])
            
            # Buscar visualizaciones
            if any(viz in source for viz in ['plt.', 'sns.', 'plotly', 'fig', 'ax']):
                analysis['visualizations'].append(i)
            
            # Buscar modelos de ML/DL
            if any(model in source for model in ['model', 'Model', 'fit(', 'train', 'predict']):
                analysis['models'].append(i)
        
        elif cell_type == 'markdown' and source.strip():
            # Buscar headers
            if source.strip().startswith('#'):
                lines = source.split('\n')
                for line in lines:
                    if line.strip().startswith('#'):
                        level = len(line) - len(line.lstrip('#'))
                        title = line.strip('#').strip()
                        
                        section_info = {
                            'level': level,
                            'title': title,
                            'cell_index': i,
                            'content': source
                        }
                        analysis['sections'].append(section_info)
                        
                        if level == 1:
                            current_section = title
                        break
        
        cell_info['section'] = current_section
        analysis['cells'].append(cell_info)
    
    return analysis

# Análisis del notebook
notebook_path = "Z:/Iris/01-Learning-Datos/Big-Data/0_Repositorios-Git-Hub/Deep-Learning/deep_learning/Practica_DL_solución.ipynb"
analysis = analyze_notebook_comprehensive(notebook_path)

# Guardar análisis
with open("notebook_analysis.json", "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print("Análisis completado y guardado en notebook_analysis.json")
print(f"Total celdas: {len(analysis['cells'])}")
print(f"Secciones encontradas: {len(analysis['sections'])}")
print(f"Funciones definidas: {len(analysis['functions'])}")
print(f"Clases definidas: {len(analysis['classes'])}")
print(f"Imports únicos: {len(set(analysis['imports']))}")
