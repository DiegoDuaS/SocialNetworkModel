import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
import networkx as nx

class SimulationAnalyzer:
    """
    Clase para análisis avanzado y visualización de simulaciones SDN.
    Implementa métricas epidemiológicas, análisis de red y visualizaciones.
    """
    
    def __init__(self, output_dir: str = "AnalysisResults"):
        """
        Inicializa el analizador.
        
        Args:
            output_dir: Directorio para guardar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Subdirectorios para organización
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "networks").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        
        # Configurar estilo de gráficas
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
    # ============================================================
    # MÉTRICAS EPIDEMIOLÓGICAS AVANZADAS
    # ============================================================
    
    def calcular_metricas_epidemiologicas(self, df: pd.DataFrame) -> Dict:
        """
        Calcula métricas epidemiológicas clave del modelo SDN.
        
        Métricas:
        - R₀ efectivo (número reproductivo básico)
        - Tiempo al pico de desinformación
        - Tasa máxima de crecimiento
        - Prevalencia máxima
        - Tiempo de duplicación inicial
        - Área bajo la curva (AUC) de desinformados
        - Tasa de ataque final
        """
        n_total = df['S'].iloc[0] + df['D'].iloc[0] + df['N'].iloc[0]
        
        # 1. Tiempo al pico de desinformación
        idx_pico = df['D'].idxmax()
        tiempo_pico = df.loc[idx_pico, 't']
        prevalencia_maxima = df.loc[idx_pico, 'D'] / n_total
        
        # 2. Tasa máxima de crecimiento (derivada máxima)
        df['tasa_cambio_D'] = df['D'].diff()
        idx_max_crecimiento = df['tasa_cambio_D'].idxmax()
        tasa_max_crecimiento = df.loc[idx_max_crecimiento, 'tasa_cambio_D']
        
        # 3. R₀ efectivo (estimación basada en fase exponencial inicial)
        # Usar primeros 10 pasos donde hay crecimiento exponencial
        fase_inicial = df[df['D'] > 0].head(10)
        if len(fase_inicial) > 3:
            log_D = np.log(fase_inicial['D'] + 1)
            t = fase_inicial['t'].values
            # Ajuste lineal: log(D) = r*t + c
            slope, intercept = np.polyfit(t, log_D, 1)
            tasa_crecimiento = slope
            # R₀ ≈ 1 + r*T_gen (asumiendo T_gen ≈ 1 paso)
            R0_estimado = 1 + tasa_crecimiento
        else:
            R0_estimado = np.nan
            tasa_crecimiento = np.nan
        
        # 4. Tiempo de duplicación inicial
        if tasa_crecimiento > 0:
            tiempo_duplicacion = np.log(2) / tasa_crecimiento
        else:
            tiempo_duplicacion = np.inf
        
        # 5. Área bajo la curva de desinformados (AUC)
        auc_desinformados = np.trapz(df['D'], df['t'])
        
        # 6. Tasa de ataque final (proporción que fue desinformada)
        # Asumiendo que la mayoría termina en N o S después de haber sido D
        desinformados_finales = df['D'].iloc[-1]
        susceptibles_finales = df['S'].iloc[-1]
        tasa_ataque = 1 - (susceptibles_finales / n_total)
        
        # 7. Duración de la epidemia (tiempo hasta < 5% del pico)
        umbral = 0.05 * df['D'].max()
        df_post_pico = df[df['t'] > tiempo_pico]
        try:
            tiempo_fin = df_post_pico[df_post_pico['D'] < umbral]['t'].iloc[0]
            duracion = tiempo_fin - df[df['D'] > 0]['t'].iloc[0]
        except:
            duracion = df['t'].iloc[-1]
        
        # 8. Velocidad de propagación promedio
        velocidad_propagacion = prevalencia_maxima / tiempo_pico if tiempo_pico > 0 else 0
        
        metricas = {
            'R0_estimado': R0_estimado,
            'tiempo_al_pico': tiempo_pico,
            'prevalencia_maxima': prevalencia_maxima,
            'tasa_max_crecimiento': tasa_max_crecimiento,
            'tasa_crecimiento_exponencial': tasa_crecimiento,
            'tiempo_duplicacion': tiempo_duplicacion,
            'auc_desinformados': auc_desinformados,
            'tasa_ataque_final': tasa_ataque,
            'duracion_epidemia': duracion,
            'velocidad_propagacion': velocidad_propagacion,
            'desinformados_finales': desinformados_finales,
            'susceptibles_finales': susceptibles_finales,
            'no_involucrados_finales': df['N'].iloc[-1]
        }
        
        return metricas
    
    def calcular_metricas_temporales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas en cada paso temporal.
        
        Returns:
            DataFrame con métricas temporales adicionales
        """
        df_metrics = df.copy()
        n_total = df['S'].iloc[0] + df['D'].iloc[0] + df['N'].iloc[0]
        
        # Proporciones
        df_metrics['prop_S'] = df['S'] / n_total
        df_metrics['prop_D'] = df['D'] / n_total
        df_metrics['prop_N'] = df['N'] / n_total
        
        # Tasas de cambio
        df_metrics['delta_S'] = df['S'].diff()
        df_metrics['delta_D'] = df['D'].diff()
        df_metrics['delta_N'] = df['N'].diff()
        
        # Flujos netos
        df_metrics['flujo_SD'] = df_metrics['delta_D'].clip(lower=0)  # S->D
        df_metrics['flujo_DS'] = -df_metrics['delta_D'].clip(upper=0)  # D->S
        
        # Momentum (segunda derivada)
        df_metrics['momentum_D'] = df_metrics['delta_D'].diff()
        
        # Volatilidad (desviación estándar móvil)
        df_metrics['volatilidad_D'] = df['D'].rolling(window=5, min_periods=1).std()
        
        return df_metrics
    
    # ============================================================
    # ANÁLISIS DE AGENTES Y RED
    # ============================================================
    
    def analizar_agentes(self, simulation) -> Dict:
        """
        Analiza características y comportamiento de agentes.
        
        Args:
            simulation: Objeto Simulation con agentes
            
        Returns:
            Diccionario con análisis de agentes
        """
        agents = simulation.agents
        
        # Estadísticas por tipo
        tipos = {'credulo': [], 'esceptico': [], 'verificador': []}
        estados = {'S': [], 'D': [], 'N': []}
        
        susceptibilidades = []
        actividades = []
        intereses = []
        
        for agent in agents.values():
            susceptibilidades.append(agent.susceptibilidad)
            actividades.append(agent.actividad)
            intereses.append(agent.interes)
            
            tipos[agent.tipo.value].append(agent)
            estados[agent.estado.value].append(agent)
        
        # Análisis por tipo
        analisis_tipos = {}
        for tipo, agentes_tipo in tipos.items():
            if len(agentes_tipo) > 0:
                analisis_tipos[tipo] = {
                    'n_agentes': len(agentes_tipo),
                    'proporcion': len(agentes_tipo) / len(agents),
                    'susceptibilidad_media': np.mean([a.susceptibilidad for a in agentes_tipo]),
                    'actividad_media': np.mean([a.actividad for a in agentes_tipo]),
                    'interes_medio': np.mean([a.interes for a in agentes_tipo]),
                    'desinformados': sum(1 for a in agentes_tipo if a.es_desinformado()),
                    'tasa_desinformacion': sum(1 for a in agentes_tipo if a.es_desinformado()) / len(agentes_tipo)
                }
        
        # Análisis por estado
        analisis_estados = {}
        for estado, agentes_estado in estados.items():
            if len(agentes_estado) > 0:
                analisis_estados[estado] = {
                    'n_agentes': len(agentes_estado),
                    'proporcion': len(agentes_estado) / len(agents),
                    'susceptibilidad_media': np.mean([a.susceptibilidad for a in agentes_estado]),
                    'actividad_media': np.mean([a.actividad for a in agentes_estado])
                }
        
        # Súper propagadores
        super_props = [a for a in agents.values() if a.es_super_propagador]
        hubs = [a for a in agents.values() if a.es_hub]
        
        analisis = {
            'n_agentes_total': len(agents),
            'distribucion_tipos': analisis_tipos,
            'distribucion_estados': analisis_estados,
            'caracteristicas_globales': {
                'susceptibilidad_media': np.mean(susceptibilidades),
                'susceptibilidad_std': np.std(susceptibilidades),
                'actividad_media': np.mean(actividades),
                'actividad_std': np.std(actividades),
                'interes_medio': np.mean(intereses),
                'interes_std': np.std(intereses)
            },
            'super_propagadores': {
                'n_super_props': len(super_props),
                'proporcion': len(super_props) / len(agents),
                'desinformados': sum(1 for a in super_props if a.es_desinformado()),
                'actividad_media': np.mean([a.actividad for a in super_props]) if super_props else 0
            },
            'hubs': {
                'n_hubs': len(hubs),
                'proporcion': len(hubs) / len(agents),
                'desinformados': sum(1 for a in hubs if a.es_desinformado())
            }
        }
        
        return analisis
    
    def analizar_red(self, network) -> Dict:
        """
        Analiza propiedades estructurales de la red.
        
        Args:
            network: Objeto SocialNetwork
            
        Returns:
            Diccionario con métricas de red
        """
        G = network.G
        
        # Métricas básicas
        metricas = network.estadisticas_red()
        
        # Centralidades
        print("Calculando centralidades...")
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        closeness = nx.closeness_centrality(G)
        
        metricas['centralidades'] = {
            'degree_media': np.mean(list(degree_centrality.values())),
            'degree_max': np.max(list(degree_centrality.values())),
            'betweenness_media': np.mean(list(betweenness.values())),
            'betweenness_max': np.max(list(betweenness.values())),
            'closeness_media': np.mean(list(closeness.values()))
        }
        
        # Asortatividad (homofilia)
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
            metricas['asortatividad'] = assortativity
        except:
            metricas['asortatividad'] = None
        
        # Comunidades
        try:
            comunidades = network.identificar_comunidades()
            n_comunidades = len(set(comunidades.values()))
            metricas['n_comunidades'] = n_comunidades
            
            # Modularidad
            G_undir = G.to_undirected()
            partition = comunidades
            modularity = nx.community.modularity(
                G_undir, 
                [set([n for n, c in partition.items() if c == i]) 
                 for i in range(n_comunidades)]
            )
            metricas['modularidad'] = modularity
        except:
            metricas['n_comunidades'] = None
            metricas['modularidad'] = None
        
        return metricas
    
    # ============================================================
    # VISUALIZACIONES
    # ============================================================
    
    def plot_evolucion_temporal(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Gráfica principal: evolución temporal de estados S-D-N.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Evolución absoluta
        ax = axes[0, 0]
        ax.plot(df['t'], df['S'], 'b-', linewidth=2, label='Susceptibles (S)')
        ax.plot(df['t'], df['D'], 'r-', linewidth=2, label='Desinformados (D)')
        ax.plot(df['t'], df['N'], 'gray', linewidth=2, label='No Involucrados (N)')
        ax.fill_between(df['t'], df['D'], alpha=0.3, color='red')
        ax.set_xlabel('Tiempo (t)')
        ax.set_ylabel('Número de agentes')
        ax.set_title('Evolución Temporal de Estados SDN', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 2. Proporciones
        ax = axes[0, 1]
        n_total = df['S'].iloc[0] + df['D'].iloc[0] + df['N'].iloc[0]
        ax.plot(df['t'], df['S']/n_total, 'b-', linewidth=2, label='S')
        ax.plot(df['t'], df['D']/n_total, 'r-', linewidth=2, label='D')
        ax.plot(df['t'], df['N']/n_total, 'gray', linewidth=2, label='N')
        ax.set_xlabel('Tiempo (t)')
        ax.set_ylabel('Proporción')
        ax.set_title('Proporciones Relativas', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Tasa de cambio de desinformados
        ax = axes[1, 0]
        delta_D = df['D'].diff()
        ax.plot(df['t'], delta_D, 'r-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(df['t'], delta_D, 0, where=(delta_D>=0), 
                        alpha=0.3, color='red', label='Crecimiento')
        ax.fill_between(df['t'], delta_D, 0, where=(delta_D<0), 
                        alpha=0.3, color='green', label='Decrecimiento')
        ax.set_xlabel('Tiempo (t)')
        ax.set_ylabel('ΔD (cambio en desinformados)')
        ax.set_title('Tasa de Cambio de Desinformación', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Viralidad y susceptibilidad media
        ax = axes[1, 1]
        ax.plot(df['t'], df['viralidad'], 'purple', linewidth=2, label='Viralidad V(t)')
        ax.set_xlabel('Tiempo (t)')
        ax.set_ylabel('Viralidad', color='purple')
        ax.tick_params(axis='y', labelcolor='purple')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(df['t'], df['avg_susceptibilidad'], 'orange', 
                linewidth=2, label='Susceptibilidad media')
        ax2.set_ylabel('Susceptibilidad media', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_title('Viralidad y Susceptibilidad Media', fontweight='bold')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'plots' / 'evolucion_temporal.png', 
                       dpi=300, bbox_inches='tight')
            print(f" Gráfica guardada: evolucion_temporal.png")
        
        plt.show()
    
    def plot_fase_espacio(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Diagrama de fase en el espacio S-D-N.
        """
        fig = plt.figure(figsize=(12, 5))
        
        # 1. Plano S-D
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(df['S'], df['D'], c=df['t'], cmap='viridis', 
                             s=30, alpha=0.6)
        ax1.plot(df['S'], df['D'], 'k-', alpha=0.3, linewidth=0.5)
        ax1.scatter(df['S'].iloc[0], df['D'].iloc[0], 
                   color='green', s=200, marker='o', 
                   edgecolors='black', linewidths=2, label='Inicio', zorder=5)
        ax1.scatter(df['S'].iloc[-1], df['D'].iloc[-1], 
                   color='red', s=200, marker='X', 
                   edgecolors='black', linewidths=2, label='Final', zorder=5)
        ax1.set_xlabel('Susceptibles (S)')
        ax1.set_ylabel('Desinformados (D)')
        ax1.set_title('Espacio de Fase S-D', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Tiempo (t)')
        
        # 2. Plano D-N
        ax2 = fig.add_subplot(122)
        scatter = ax2.scatter(df['D'], df['N'], c=df['t'], cmap='plasma', 
                             s=30, alpha=0.6)
        ax2.plot(df['D'], df['N'], 'k-', alpha=0.3, linewidth=0.5)
        ax2.scatter(df['D'].iloc[0], df['N'].iloc[0], 
                   color='green', s=200, marker='o', 
                   edgecolors='black', linewidths=2, label='Inicio', zorder=5)
        ax2.scatter(df['D'].iloc[-1], df['N'].iloc[-1], 
                   color='red', s=200, marker='X', 
                   edgecolors='black', linewidths=2, label='Final', zorder=5)
        ax2.set_xlabel('Desinformados (D)')
        ax2.set_ylabel('No Involucrados (N)')
        ax2.set_title('Espacio de Fase D-N', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Tiempo (t)')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'plots' / 'diagrama_fase.png', 
                       dpi=300, bbox_inches='tight')
            print(f" Gráfica guardada: diagrama_fase.png")
        
        plt.show()
    
    def plot_distribucion_caracteristicas(self, simulation, save: bool = True) -> None:
        """
        Distribuciones de características de agentes.
        """
        agents = simulation.agents
        
        # Extraer datos
        susceptibilidades = [a.susceptibilidad for a in agents.values()]
        actividades = [a.actividad for a in agents.values()]
        intereses = [a.interes for a in agents.values()]
        tipos = [a.tipo.value for a in agents.values()]
        estados = [a.estado.value for a in agents.values()]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Susceptibilidad
        ax = axes[0, 0]
        ax.hist(susceptibilidades, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(susceptibilidades), color='red', linestyle='--', 
                  linewidth=2, label=f'Media: {np.mean(susceptibilidades):.3f}')
        ax.set_xlabel('Susceptibilidad (s_i)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Susceptibilidad', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Actividad (escala log)
        ax = axes[0, 1]
        ax.hist(actividades, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(actividades), color='red', linestyle='--', 
                  linewidth=2, label=f'Media: {np.mean(actividades):.3f}')
        ax.set_xlabel('Actividad (a_i)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Actividad', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Interés
        ax = axes[0, 2]
        ax.hist(intereses, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.mean(intereses), color='red', linestyle='--', 
                  linewidth=2, label=f'Media: {np.mean(intereses):.3f}')
        ax.set_xlabel('Interés (r_i)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Interés', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Tipos de agentes
        ax = axes[1, 0]
        tipo_counts = pd.Series(tipos).value_counts()
        colors = {'credulo': 'red', 'esceptico': 'blue', 'verificador': 'green'}
        bars = ax.bar(tipo_counts.index, tipo_counts.values, 
                     color=[colors[t] for t in tipo_counts.index], 
                     alpha=0.7, edgecolor='black')
        ax.set_ylabel('Número de agentes')
        ax.set_title('Distribución de Tipos de Agentes', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(agents)*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        # 5. Estados actuales
        ax = axes[1, 1]
        estado_counts = pd.Series(estados).value_counts()
        colors_estado = {'S': 'blue', 'D': 'red', 'N': 'gray'}
        bars = ax.bar(estado_counts.index, estado_counts.values,
                     color=[colors_estado[e] for e in estado_counts.index],
                     alpha=0.7, edgecolor='black')
        ax.set_ylabel('Número de agentes')
        ax.set_title('Distribución de Estados Actuales', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(agents)*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        # 6. Correlación Susceptibilidad vs Actividad por tipo
        ax = axes[1, 2]
        for tipo in ['credulo', 'esceptico', 'verificador']:
            agentes_tipo = [a for a in agents.values() if a.tipo.value == tipo]
            if agentes_tipo:
                s = [a.susceptibilidad for a in agentes_tipo]
                a = [a.actividad for a in agentes_tipo]
                ax.scatter(s, a, alpha=0.6, s=50, label=tipo.capitalize())
        ax.set_xlabel('Susceptibilidad')
        ax.set_ylabel('Actividad')
        ax.set_title('Susceptibilidad vs Actividad por Tipo', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'plots' / 'distribucion_agentes.png', 
                       dpi=300, bbox_inches='tight')
            print(f" Gráfica guardada: distribucion_agentes.png")
        
        plt.show()
    
    def plot_analisis_superpropagadores(self, simulation, df: pd.DataFrame, 
                                       save: bool = True) -> None:
        """
        Análisis específico de súper propagadores y hubs.
        """
        agents = simulation.agents
        
        # Clasificar agentes
        super_props = [a for a in agents.values() if a.es_super_propagador]
        hubs = [a for a in agents.values() if a.es_hub]
        regulares = [a for a in agents.values() 
                    if not (a.es_super_propagador or a.es_hub)]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Comparación de actividad
        ax = axes[0, 0]
        data_actividad = []
        labels = []
        if regulares:
            data_actividad.append([a.actividad for a in regulares])
            labels.append('Regulares')
        if hubs:
            data_actividad.append([a.actividad for a in hubs])
            labels.append('Hubs')
        if super_props:
            data_actividad.append([a.actividad for a in super_props])
            labels.append('Súper Prop.')
        
        bp = ax.boxplot(data_actividad, labels=labels, patch_artist=True)
        colors = ['lightblue', 'orange', 'red']
        for patch, color in zip(bp['boxes'], colors[:len(data_actividad)]):
            patch.set_facecolor(color)
        ax.set_ylabel('Actividad')
        ax.set_title('Comparación de Actividad por Tipo', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Tasa de desinformación por tipo
        ax = axes[0, 1]
        categorias = ['Regulares', 'Hubs', 'Súper Prop.']
        tasas = []
        for grupo in [regulares, hubs, super_props]:
            if grupo:
                tasa = sum(1 for a in grupo if a.es_desinformado()) / len(grupo)
                tasas.append(tasa * 100)
            else:
                tasas.append(0)
        
        bars = ax.bar(categorias, tasas, color=['lightblue', 'orange', 'red'], 
                     alpha=0.7, edgecolor='black')
        ax.set_ylabel('% Desinformados')
        ax.set_title('Tasa de Desinformación por Tipo', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, tasa in zip(bars, tasas):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{tasa:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Distribución de susceptibilidad por tipo
        ax = axes[1, 0]
        data_susc = []
        labels_susc = []
        if regulares:
            data_susc.append([a.susceptibilidad for a in regulares])
            labels_susc.append('Regulares')
        if hubs:
            data_susc.append([a.susceptibilidad for a in hubs])
            labels_susc.append('Hubs')
        if super_props:
            data_susc.append([a.susceptibilidad for a in super_props])
            labels_susc.append('Súper Prop.')
        
        bp = ax.boxplot(data_susc, labels=labels_susc, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data_susc)]):
            patch.set_facecolor(color)
        ax.set_ylabel('Susceptibilidad')
        ax.set_title('Comparación de Susceptibilidad', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Contribución acumulada a la viralidad
        ax = axes[1, 1]
        # Calcular contribución de cada tipo a la viralidad total
        actividad_total = sum(a.obtener_actividad_efectiva() for a in agents.values())
        contribuciones = {}
        for nombre, grupo in [('Regulares', regulares), ('Hubs', hubs), 
                             ('Súper Prop.', super_props)]:
            if grupo and actividad_total > 0:
                contrib = sum(a.obtener_actividad_efectiva() for a in grupo) / actividad_total
                contribuciones[nombre] = contrib * 100
            else:
                contribuciones[nombre] = 0
        
        wedges, texts, autotexts = ax.pie(contribuciones.values(), 
                                           labels=contribuciones.keys(),
                                           autopct='%1.1f%%',
                                           colors=['lightblue', 'orange', 'red'],
                                           startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Contribución a la Actividad Total', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'plots' / 'analisis_superpropagadores.png', 
                       dpi=300, bbox_inches='tight')
            print(f" Gráfica guardada: analisis_superpropagadores.png")
        
        plt.show()
    
    def plot_red_con_estados(self, network, simulation, save: bool = True,
                            layout: str = 'spring') -> None:
        """
        Visualiza la red coloreada por estados actuales de los agentes.
        """
        G = network.G
        agents = simulation.agents
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=network.seed, k=0.5, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # 1. Coloreado por estado
        ax = axes[0]
        color_map = {'S': 'blue', 'D': 'red', 'N': 'gray'}
        node_colors = [color_map[agents[node].estado.value] for node in G.nodes()]
        node_sizes = [50 + 10 * G.degree(node) for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=False, ax=ax)
        
        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Susceptibles'),
            Patch(facecolor='red', alpha=0.7, label='Desinformados'),
            Patch(facecolor='gray', alpha=0.7, label='No Involucrados')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        ax.set_title('Red Social por Estado (SDN)', fontweight='bold', fontsize=14)
        ax.axis('off')
        
        # 2. Destacar súper propagadores y hubs
        ax = axes[1]
        
        # Colorear por tipo especial
        node_colors_sp = []
        for node in G.nodes():
            agent = agents[node]
            if agent.es_super_propagador:
                node_colors_sp.append('red')
            elif agent.es_hub:
                node_colors_sp.append('orange')
            else:
                node_colors_sp.append('lightblue')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors_sp, 
                              node_size=node_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=False, ax=ax)
        
        # Leyenda
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='Regulares'),
            Patch(facecolor='orange', alpha=0.7, label='Hubs'),
            Patch(facecolor='red', alpha=0.7, label='Súper Propagadores')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        ax.set_title('Red Social por Tipo de Agente', fontweight='bold', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'networks' / 'red_estados.png', 
                       dpi=300, bbox_inches='tight')
            print(f" Gráfica guardada: red_estados.png")
        
        plt.show()
    
    def plot_mapa_calor_influencia(self, network, simulation, save: bool = True) -> None:
        """
        Mapa de calor de la matriz de influencia (pesos * actividad).
        """
        G = network.G
        agents = simulation.agents
        
        # Construir matriz de influencia
        n = len(agents)
        matriz_influencia = np.zeros((n, n))
        
        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        for i, node_i in enumerate(node_list):
            for node_j in G.successors(node_i):
                j = node_to_idx[node_j]
                peso = network.obtener_peso(node_i, node_j)
                actividad = agents[node_j].obtener_actividad_efectiva()
                matriz_influencia[i, j] = peso * actividad
        
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 1. Mapa de calor completo (muestra limitada si es muy grande)
        ax = axes[0]
        if n <= 100:
            im = ax.imshow(matriz_influencia, cmap='YlOrRd', aspect='auto')
            ax.set_xlabel('Nodo j (influenciado)')
            ax.set_ylabel('Nodo i (influenciador)')
        else:
            # Mostrar solo primeros 100x100
            im = ax.imshow(matriz_influencia[:100, :100], cmap='YlOrRd', aspect='auto')
            ax.set_xlabel('Nodo j (influenciado) - Primeros 100')
            ax.set_ylabel('Nodo i (influenciador) - Primeros 100')
        
        ax.set_title('Matriz de Influencia (peso × actividad)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Influencia')
        
        # 2. Distribución de influencias
        ax = axes[1]
        influencias_no_cero = matriz_influencia[matriz_influencia > 0].flatten()
        ax.hist(influencias_no_cero, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Influencia (peso × actividad)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Influencias en la Red', fontweight='bold')
        ax.axvline(np.mean(influencias_no_cero), color='red', linestyle='--',
                  linewidth=2, label=f'Media: {np.mean(influencias_no_cero):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'networks' / 'mapa_calor_influencia.png', 
                       dpi=300, bbox_inches='tight')
            print(f" Gráfica guardada: mapa_calor_influencia.png")
        
        plt.show()
    
    # ============================================================
    # ANÁLISIS COMPARATIVO Y SENSIBILIDAD
    # ============================================================
    
    def comparar_simulaciones(self, resultados: List[Tuple[str, pd.DataFrame]], 
                             save: bool = True) -> None:
        """
        Compara múltiples simulaciones con diferentes parámetros.
        
        Args:
            resultados: Lista de tuplas (nombre, dataframe)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(resultados)))
        
        # 1. Evolución de desinformados
        ax = axes[0, 0]
        for (nombre, df), color in zip(resultados, colors):
            ax.plot(df['t'], df['D'], linewidth=2, label=nombre, color=color)
        ax.set_xlabel('Tiempo (t)')
        ax.set_ylabel('Desinformados')
        ax.set_title('Comparación: Evolución de Desinformados', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Pico de desinformación
        ax = axes[0, 1]
        nombres = [nombre for nombre, _ in resultados]
        picos = [df['D'].max() for _, df in resultados]
        tiempos_pico = [df.loc[df['D'].idxmax(), 't'] for _, df in resultados]
        
        x = np.arange(len(nombres))
        width = 0.35
        ax.bar(x - width/2, picos, width, label='Pico máximo', alpha=0.7, color=colors)
        ax.bar(x + width/2, tiempos_pico, width, label='Tiempo al pico', alpha=0.7)
        ax.set_ylabel('Valor')
        ax.set_title('Comparación: Pico y Tiempo', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Viralidad
        ax = axes[1, 0]
        for (nombre, df), color in zip(resultados, colors):
            ax.plot(df['t'], df['viralidad'], linewidth=2, label=nombre, color=color)
        ax.set_xlabel('Tiempo (t)')
        ax.set_ylabel('Viralidad V(t)')
        ax.set_title('Comparación: Viralidad', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Estado final
        ax = axes[1, 1]
        estados_finales = {'S': [], 'D': [], 'N': []}
        for nombre, df in resultados:
            n_total = df['S'].iloc[0] + df['D'].iloc[0] + df['N'].iloc[0]
            estados_finales['S'].append(df['S'].iloc[-1] / n_total * 100)
            estados_finales['D'].append(df['D'].iloc[-1] / n_total * 100)
            estados_finales['N'].append(df['N'].iloc[-1] / n_total * 100)
        
        x = np.arange(len(nombres))
        width = 0.25
        ax.bar(x - width, estados_finales['S'], width, label='Susceptibles', 
              color='blue', alpha=0.7)
        ax.bar(x, estados_finales['D'], width, label='Desinformados', 
              color='red', alpha=0.7)
        ax.bar(x + width, estados_finales['N'], width, label='No Involucrados', 
              color='gray', alpha=0.7)
        ax.set_ylabel('% del total')
        ax.set_title('Comparación: Estado Final', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'plots' / 'comparacion_simulaciones.png', 
                       dpi=300, bbox_inches='tight')
            print(f" Gráfica guardada: comparacion_simulaciones.png")
        
        plt.show()
    
    def analisis_sensibilidad_parametros(self, simulation_func, 
                                        parametro: str,
                                        valores: List[float],
                                        params_base: Dict,
                                        save: bool = True) -> pd.DataFrame:
        """
        Análisis de sensibilidad variando un parámetro.
        
        Args:
            simulation_func: Función que ejecuta la simulación
            parametro: Nombre del parámetro a variar
            valores: Lista de valores a probar
            params_base: Diccionario con parámetros base
            
        Returns:
            DataFrame con resultados del análisis
        """
        resultados = []
        
        print(f"\nAnálisis de sensibilidad: {parametro}")
        print("-" * 60)
        
        for valor in valores:
            params = params_base.copy()
            params[parametro] = valor
            
            print(f"  Probando {parametro}={valor:.3f}...")
            df = simulation_func(**params)
            
            metricas = self.calcular_metricas_epidemiologicas(df)
            metricas[parametro] = valor
            resultados.append(metricas)
        
        df_sensibilidad = pd.DataFrame(resultados)
        
        # Visualización
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metricas_clave = [
            ('R0_estimado', 'R₀ Estimado'),
            ('prevalencia_maxima', 'Prevalencia Máxima'),
            ('tiempo_al_pico', 'Tiempo al Pico'),
            ('tasa_ataque_final', 'Tasa de Ataque Final'),
            ('duracion_epidemia', 'Duración de Epidemia'),
            ('velocidad_propagacion', 'Velocidad de Propagación')
        ]
        
        for idx, (metrica, titulo) in enumerate(metricas_clave):
            ax = axes[idx]
            ax.plot(df_sensibilidad[parametro], df_sensibilidad[metrica], 
                   'o-', linewidth=2, markersize=8, color='steelblue')
            ax.set_xlabel(f'{parametro}')
            ax.set_ylabel(titulo)
            ax.set_title(f'{titulo} vs {parametro}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'plots' / f'sensibilidad_{parametro}.png', 
                       dpi=300, bbox_inches='tight')
            df_sensibilidad.to_csv(
                self.output_dir / 'tables' / f'sensibilidad_{parametro}.csv', 
                index=False
            )
            print(f" Análisis guardado: sensibilidad_{parametro}.png")
        
        plt.show()
        
        return df_sensibilidad
    
    # ============================================================
    # GENERACIÓN DE REPORTES
    # ============================================================
    
    def generar_tabla_metricas(self, metricas: Dict, save: bool = True) -> pd.DataFrame:
        """
        Genera tabla formateada con métricas epidemiológicas.
        """
        # Formatear métricas
        tabla_data = []
        for key, value in metricas.items():
            if isinstance(value, (int, float, np.number)):
                if 'tiempo' in key or 'duracion' in key:
                    valor_fmt = f"{value:.2f} pasos"
                elif 'tasa' in key or 'proporcion' in key or 'prevalencia' in key:
                    valor_fmt = f"{value*100:.2f}%"
                elif 'R0' in key:
                    valor_fmt = f"{value:.3f}"
                else:
                    valor_fmt = f"{value:.4f}"
                
                tabla_data.append({
                    'Métrica': key.replace('_', ' ').title(),
                    'Valor': valor_fmt
                })
        
        df_tabla = pd.DataFrame(tabla_data)
        
        if save:
            df_tabla.to_csv(self.output_dir / 'tables' / 'metricas_epidemiologicas.csv', 
                           index=False)
            print(f"Tabla guardada: metricas_epidemiologicas.csv")
        
        return df_tabla
    
    def generar_reporte_completo(self, simulation, df: pd.DataFrame, 
                                network, save: bool = True) -> Dict:
        """
        Genera reporte completo con todas las métricas y análisis.
        
        Returns:
            Diccionario con todos los resultados del análisis
        """
        print("\n" + "="*70)
        print("GENERANDO REPORTE COMPLETO DE ANÁLISIS")
        print("="*70)
        
        reporte = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parametros_simulacion': {},
            'metricas_epidemiologicas': {},
            'analisis_agentes': {},
            'analisis_red': {}
        }
        
        # 1. Métricas epidemiológicas
        print("\n1. Calculando métricas epidemiológicas...")
        metricas_epi = self.calcular_metricas_epidemiologicas(df)
        reporte['metricas_epidemiologicas'] = metricas_epi
        
        # 2. Análisis de agentes
        print("2. Analizando agentes...")
        analisis_agentes = self.analizar_agentes(simulation)
        reporte['analisis_agentes'] = analisis_agentes
        
        # 3. Análisis de red
        print("3. Analizando estructura de red...")
        analisis_red = self.analizar_red(network)
        reporte['analisis_red'] = analisis_red
        
        # 4. Generar visualizaciones
        print("\n4. Generando visualizaciones...")
        print("   - Evolución temporal...")
        self.plot_evolucion_temporal(df, save=save)
        
        print("   - Diagrama de fase...")
        self.plot_fase_espacio(df, save=save)
        
        print("   - Distribución de características...")
        self.plot_distribucion_caracteristicas(simulation, save=save)
        
        print("   - Análisis de súper propagadores...")
        self.plot_analisis_superpropagadores(simulation, df, save=save)
        
        print("   - Visualización de red...")
        self.plot_red_con_estados(network, simulation, save=save)
        
        print("   - Mapa de calor de influencia...")
        self.plot_mapa_calor_influencia(network, simulation, save=save)
        
        # 5. Generar tablas
        print("\n5. Generando tablas...")
        self.generar_tabla_metricas(metricas_epi, save=save)
        
        # 6. Guardar reporte completo como JSON
        if save:
            # Convertir numpy types a tipos nativos de Python
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            reporte_serializable = convert_types(reporte)
            
            with open(self.output_dir / 'reporte_completo.json', 'w') as f:
                json.dump(reporte_serializable, f, indent=2)
            
            print(f"\n Reporte completo guardado: reporte_completo.json")
        
        # 7. Resumen en consola
        self._imprimir_resumen(reporte)
        
        print("\n" + "="*70)
        print("ANÁLISIS COMPLETADO")
        print(f"Resultados guardados en: {self.output_dir}")
        print("="*70 + "\n")
        
        return reporte
    
    def _imprimir_resumen(self, reporte: Dict) -> None:
        """Imprime resumen legible del reporte."""
        print("\n" + "-"*70)
        print("RESUMEN DE RESULTADOS")
        print("-"*70)
        
        metricas = reporte['metricas_epidemiologicas']
        print("\n MÉTRICAS EPIDEMIOLÓGICAS CLAVE:")
        print(f"   R₀ estimado: {metricas['R0_estimado']:.3f}")
        print(f"   Prevalencia máxima: {metricas['prevalencia_maxima']*100:.2f}%")
        print(f"   Tiempo al pico: {metricas['tiempo_al_pico']:.1f} pasos")
        print(f"   Tasa de ataque final: {metricas['tasa_ataque_final']*100:.2f}%")
        print(f"   Duración: {metricas['duracion_epidemia']:.1f} pasos")
        
        agentes = reporte['analisis_agentes']
        print("\n ANÁLISIS DE AGENTES:")
        print(f"   Total: {agentes['n_agentes_total']}")
        print(f"   Súper propagadores: {agentes['super_propagadores']['n_super_props']} "
              f"({agentes['super_propagadores']['proporcion']*100:.1f}%)")
        print(f"   Hubs: {agentes['hubs']['n_hubs']} "
              f"({agentes['hubs']['proporcion']*100:.1f}%)")
        
        red = reporte['analisis_red']
        print("\n  ESTRUCTURA DE RED:")
        print(f"   Nodos: {red['n_nodos']}, Aristas: {red['n_aristas']}")
        print(f"   Densidad: {red['densidad']:.4f}")
        print(f"   Clustering: {red['coeficiente_clustering']:.4f}")
        if 'camino_promedio' in red and isinstance(red['camino_promedio'], (int, float)):
            print(f"   Camino promedio: {red['camino_promedio']:.2f}")
        if red.get('n_comunidades'):
            print(f"   Comunidades: {red['n_comunidades']}")
        
        print("-"*70)