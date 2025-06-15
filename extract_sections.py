#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def extract_key_sections(notebook_path, max_chars_per_section=3000):
    """Extrae las secciones clave del notebook para análisis"""
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return None
    
    key_sections = {}
    current_section = "Inicio"
    section_content = []
    
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']) if cell['source'] else ""
        
        # Detectar nueva sección
        if cell['cell_type'] == 'markdown' and source.strip().startswith('#'):
            # Guardar sección anterior si tiene contenido
            if section_content:
                key_sections[current_section] = '\n'.join(section_content[:10])  # Limitar contenido
            
            # Nueva sección
            lines = source.split('\n')
            for line in lines:
                if line.strip().startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    if level <= 2:  # Solo niveles 1 y 2
                        current_section = line.strip('#').strip()
                        section_content = [source]
                        break
        else:
            # Agregar contenido a la sección actual
            if len('\n'.join(section_content)) < max_chars_per_section:
                section_content.append(source[:500])  # Limitar por celda
    
    # Guardar última sección
    if section_content:
        key_sections[current_section] = '\n'.join(section_content[:10])
    
    return key_sections

# Extraer secciones clave
notebook_path = "Z:/Iris/01-Learning-Datos/Big-Data/0_Repositorios-Git-Hub/Deep-Learning/deep_learning/Practica_DL_solución.ipynb"
sections = extract_key_sections(notebook_path)

if sections:
    print("SECCIONES EXTRAÍDAS:")
    print("=" * 50)
    for section_name, content in sections.items():
        print(f"\n### {section_name}")
        print("-" * 30)
        print(content[:800] + "..." if len(content) > 800 else content)
        print()
else:
    print("No se pudieron extraer las secciones")
