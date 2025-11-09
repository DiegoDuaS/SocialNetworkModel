import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
from pathlib import Path

from SocialNetwork import SocialNetwork
from Simulation import Simulation, ejecutar_simulacion_completa
from Analyzer import SimulationAnalyzer


class ExperimentoComparativo:
    """
    Clase para ejecutar múltiples simulaciones y comparar resultados.
    """
    
    def __init__(self, output_dir: str = "Experimentos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.resultados = []
    
    def experimento_variacion_parametro(self, 
                                       parametro: str,
                                       valores: List[float],
                                       params_base_red: Dict,
                                       params_base_sim: Dict,
                                       n_replicas: int = 3) -> pd.DataFrame:
        """
        Experimenta variando un parámetro específico.
        
        Args:
            parametro: Nombre del parámetro a variar (debe estar en params_base_sim)
            valores: Lista de valores a probar
            params_base_red: Parámetros base para la red
            params_base_sim: Parámetros base para simulación
            n_replicas: Número de réplicas por valor (para promediar)
            
        Returns:
            DataFrame con resultados agregados
        """
        print(f"\n{'='*70}")
        print(f"EXPERIMENTO: Variación de {parametro}")
        print(f"Valores: {valores}")
        print(f"Réplicas por valor: {n_replicas}")
        print(f"{'='*70}\n")
        
        resultados_exp = []
        
        for valor in valores:
            print(f"\n--- Probando {parametro} = {valor} ---")
            
            metricas_replicas = []
            
            for replica in range(n_replicas):
                # Actualizar parámetro
                params_sim = params_base_sim.copy()
                params_sim[parametro] = valor
                
                # Cambiar seed para cada réplica
                params_red = params_base_red.copy()
                params_red['seed'] = params_base_red['seed'] + replica
                
                print(f"  Réplica {replica + 1}/{n_replicas}...")
                
                # Ejecutar simulación
                red = SocialNetwork(**params_red)
                red.asignar_super_propagadores()
                sim = Simulation(network=red, seed=params_red['seed'])
                df = sim.run(**params_sim, verbose=False)
                
                # Calcular métricas
                analyzer = SimulationAnalyzer(output_dir=str(self.output_dir / "temp"))
                metricas = analyzer.calcular_metricas_epidemiologicas(df)
                metricas[parametro] = valor
                metricas['replica'] = replica
                
                metricas_replicas.append(metricas)
            
            # Promediar sobre réplicas
            df_replicas = pd.DataFrame(metricas_replicas)
            metricas_promedio = df_replicas.mean(numeric_only=True).to_dict()
            metricas_promedio[parametro] = valor
            metricas_promedio['n_replicas'] = n_replicas
            
            # Calcular desviaciones estándar
            for col in df_replicas.select_dtypes(include=[np.number]).columns:
                if col not in [parametro, 'replica']:
                    metricas_promedio[f'{col}_std'] = df_replicas[col].std()
            
            resultados_exp.append(metricas_promedio)
            
            print(f"   Completado {parametro}={valor}")
            print(f"    R0 promedio: {metricas_promedio['R0_estimado']:.3f}")
            print(f"    Prevalencia máxima: {metricas_promedio['prevalencia_maxima']*100:.2f}%")
        
        df_resultados = pd.DataFrame(resultados_exp)
        
        # Guardar resultados
        filename = self.output_dir / f"sensibilidad_{parametro}.csv"
        df_resultados.to_csv(filename, index=False)
        print(f"\n Resultados guardados: {filename}")
        
        # Visualizar
        self._plot_sensibilidad(df_resultados, parametro)
        
        return df_resultados
    
    def _plot_sensibilidad(self, df: pd.DataFrame, parametro: str):
        """Genera gráficas de sensibilidad."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metricas_clave = [
            ('R0_estimado', 'R₀ Estimado', 'blue'),
            ('prevalencia_maxima', 'Prevalencia Máxima', 'red'),
            ('tiempo_al_pico', 'Tiempo al Pico', 'green'),
            ('tasa_ataque_final', 'Tasa de Ataque Final', 'purple'),
            ('duracion_epidemia', 'Duración', 'orange'),
            ('velocidad_propagacion', 'Velocidad de Propagación', 'brown')
        ]
        
        for idx, (metrica, titulo, color) in enumerate(metricas_clave):
            ax = axes[idx]
            
            # Gráfica principal
            ax.plot(df[parametro], df[metrica], 'o-', 
                   linewidth=2, markersize=8, color=color, label='Promedio')
            
            # Barras de error si hay desviación estándar
            if f'{metrica}_std' in df.columns:
                ax.fill_between(df[parametro],
                               df[metrica] - df[f'{metrica}_std'],
                               df[metrica] + df[f'{metrica}_std'],
                               alpha=0.3, color=color)
            
            ax.set_xlabel(f'{parametro}', fontsize=11)
            ax.set_ylabel(titulo, fontsize=11)
            ax.set_title(f'{titulo} vs {parametro}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        filename = self.output_dir / f"sensibilidad_{parametro}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Gráfica guardada: {filename}")
        plt.show()
    
    def experimento_tipos_red(self, 
                             params_sim: Dict,
                             n_nodos: int = 400) -> pd.DataFrame:
        """
        Compara diferentes topologías de red.
        
        Configuraciones:
        - Small-world puro (alta conexión local)
        - Scale-free puro (muchos hubs)
        - Híbrido balanceado
        - Red aleatoria (baseline)
        """
        print(f"\n{'='*70}")
        print("EXPERIMENTO: Comparación de Topologías de Red")
        print(f"{'='*70}\n")
        
        configuraciones = [
            {
                'nombre': 'Small-world',
                'params': {
                    'n_nodos': n_nodos,
                    'k_vecinos': 8,
                    'p_rewire': 0.1,  # Pocas reconexiones
                    'm_hubs': 1,
                    'proporcion_hubs': 0.02,  # Pocos hubs
                    'seed': 42
                }
            },
            {
                'nombre': 'Scale-free',
                'params': {
                    'n_nodos': n_nodos,
                    'k_vecinos': 4,
                    'p_rewire': 0.5,  # Muchas reconexiones
                    'm_hubs': 5,
                    'proporcion_hubs': 0.15,  # Muchos hubs
                    'seed': 42
                }
            },
            {
                'nombre': 'Híbrido',
                'params': {
                    'n_nodos': n_nodos,
                    'k_vecinos': 6,
                    'p_rewire': 0.2,
                    'm_hubs': 3,
                    'proporcion_hubs': 0.08,
                    'seed': 42
                }
            },
            {
                'nombre': 'Aleatoria',
                'params': {
                    'n_nodos': n_nodos,
                    'k_vecinos': 6,
                    'p_rewire': 0.8,  # Muy aleatoria
                    'm_hubs': 2,
                    'proporcion_hubs': 0.05,
                    'seed': 42
                }
            }
        ]
        
        resultados = []
        series_temporales = []
        
        for config in configuraciones:
            print(f"\n--- Topología: {config['nombre']} ---")
            
            # Crear red
            red = SocialNetwork(**config['params'])
            red.asignar_super_propagadores()
            
            # Métricas de red
            stats_red = red.estadisticas_red()
            
            # Ejecutar simulación
            sim = Simulation(network=red, seed=42)
            df = sim.run(**params_sim, verbose=False)
            
            # Análisis
            analyzer = SimulationAnalyzer(output_dir=str(self.output_dir / "temp"))
            metricas_epi = analyzer.calcular_metricas_epidemiologicas(df)
            
            # Combinar resultados
            resultado = {
                'topologia': config['nombre'],
                **metricas_epi,
                'densidad_red': stats_red['densidad'],
                'clustering': stats_red['coeficiente_clustering'],
                'grado_in_promedio': stats_red['grado_in_promedio']
            }
            resultados.append(resultado)
            
            # Guardar serie temporal
            df['topologia'] = config['nombre']
            series_temporales.append(df)
            
            print(f"   Completado")
            print(f"    R0: {metricas_epi['R0_estimado']:.3f}")
            print(f"    Prevalencia máxima: {metricas_epi['prevalencia_maxima']*100:.2f}%")
            print(f"    Clustering: {stats_red['coeficiente_clustering']:.3f}")
        
        df_resultados = pd.DataFrame(resultados)
        df_series = pd.concat(series_temporales, ignore_index=True)
        
        # Guardar
        df_resultados.to_csv(self.output_dir / "comparacion_topologias.csv", index=False)
        df_series.to_csv(self.output_dir / "series_topologias.csv", index=False)
        
        # Visualizar
        self._plot_comparacion_topologias(df_resultados, df_series)
        
        return df_resultados
    
    def _plot_comparacion_topologias(self, df_resultados: pd.DataFrame, 
                                     df_series: pd.DataFrame):
        """Visualiza comparación de topologías."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        topologias = df_resultados['topologia'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(topologias)))
        
        # 1. Evolución temporal de desinformados
        ax = axes[0, 0]
        for topo, color in zip(topologias, colors):
            df_topo = df_series[df_series['topologia'] == topo]
            ax.plot(df_topo['t'], df_topo['D'], linewidth=2.5, 
                   label=topo, color=color)
        ax.set_xlabel('Tiempo (t)', fontsize=12)
        ax.set_ylabel('Desinformados', fontsize=12)
        ax.set_title('Evolución de Desinformación por Topología', fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2. Métricas epidemiológicas
        ax = axes[0, 1]
        metricas = ['R0_estimado', 'prevalencia_maxima', 'tasa_ataque_final']
        x = np.arange(len(topologias))
        width = 0.25
        
        for i, metrica in enumerate(metricas):
            valores = df_resultados[metrica].values
            if 'tasa' in metrica or 'prevalencia' in metrica:
                valores = valores * 100
            ax.bar(x + i*width, valores, width, label=metrica.replace('_', ' ').title(),
                  alpha=0.8)
        
        ax.set_ylabel('Valor', fontsize=12)
        ax.set_title('Métricas Epidemiológicas por Topología', fontweight='bold', fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(topologias, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Propiedades estructurales de red
        ax = axes[1, 0]
        propiedades = ['densidad_red', 'clustering', 'grado_in_promedio']
        x = np.arange(len(topologias))
        width = 0.25
        
        for i, prop in enumerate(propiedades):
            ax.bar(x + i*width, df_resultados[prop].values, width,
                  label=prop.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_ylabel('Valor', fontsize=12)
        ax.set_title('Propiedades Estructurales de Red', fontweight='bold', fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(topologias, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Correlación estructura-dinámica
        ax = axes[1, 1]
        ax.scatter(df_resultados['clustering'], df_resultados['R0_estimado'],
                  s=200, c=range(len(topologias)), cmap='Set2', alpha=0.7,
                  edgecolors='black', linewidths=2)
        
        for i, topo in enumerate(topologias):
            row = df_resultados[df_resultados['topologia'] == topo].iloc[0]
            ax.annotate(topo, (row['clustering'], row['R0_estimado']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Clustering Coefficient', fontsize=12)
        ax.set_ylabel('R₀ Estimado', fontsize=12)
        ax.set_title('Correlación: Estructura de Red vs Propagación', 
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparacion_topologias.png", 
                   dpi=300, bbox_inches='tight')
        print(f" Gráfica guardada: comparacion_topologias.png")
        plt.show()
    
    def experimento_intervencion(self,
                                params_base_red: Dict,
                                params_base_sim: Dict) -> pd.DataFrame:
        """
        Simula diferentes estrategias de intervención.
        
        Estrategias:
        1. Sin intervención (baseline)
        2. Aumentar corrección (fact-checking)
        3. Reducir susceptibilidad (educación)
        4. Reducir actividad de súper propagadores (moderación)
        """
        print(f"\n{'='*70}")
        print("EXPERIMENTO: Estrategias de Intervención")
        print(f"{'='*70}\n")
        
        estrategias = [
            {
                'nombre': 'Sin intervención',
                'params': params_base_sim.copy()
            },
            {
                'nombre': 'Fact-checking intenso',
                'params': {**params_base_sim, 'gamma0': 0.35, 'gamma1': 0.08}
            },
            {
                'nombre': 'Educación mediática',
                'params': {**params_base_sim, 'beta0': 0.05, 'reinforcement_rate': 0.005}
            },
            {
                'nombre': 'Moderación de influencers',
                'params': {**params_base_sim, 'beta1': 0.2}
            },
            {
                'nombre': 'Intervención combinada',
                'params': {**params_base_sim, 'gamma0': 0.30, 'beta0': 0.06, 'beta1': 0.25}
            }
        ]
        
        resultados = []
        series_temporales = []
        
        for estrategia in estrategias:
            print(f"\n--- Estrategia: {estrategia['nombre']} ---")
            
            # Crear red (misma para todas)
            red = SocialNetwork(**params_base_red)
            red.asignar_super_propagadores()
            
            # Ejecutar simulación
            sim = Simulation(network=red, seed=params_base_red['seed'])
            df = sim.run(**estrategia['params'], verbose=False)
            
            # Análisis
            analyzer = SimulationAnalyzer(output_dir=str(self.output_dir / "temp"))
            metricas = analyzer.calcular_metricas_epidemiologicas(df)
            
            resultado = {
                'estrategia': estrategia['nombre'],
                **metricas
            }
            resultados.append(resultado)
            
            df['estrategia'] = estrategia['nombre']
            series_temporales.append(df)
            
            print(f"   Completado")
            print(f"    Reducción de prevalencia: "
                  f"{(1 - metricas['prevalencia_maxima']/resultados[0]['prevalencia_maxima'])*100:.1f}%")
        
        df_resultados = pd.DataFrame(resultados)
        df_series = pd.concat(series_temporales, ignore_index=True)
        
        # Guardar
        df_resultados.to_csv(self.output_dir / "comparacion_intervenciones.csv", index=False)
        df_series.to_csv(self.output_dir / "series_intervenciones.csv", index=False)
        
        # Visualizar
        self._plot_intervenciones(df_resultados, df_series)
        
        return df_resultados
    
    def _plot_intervenciones(self, df_resultados: pd.DataFrame, 
                            df_series: pd.DataFrame):
        """Visualiza comparación de intervenciones."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        estrategias = df_resultados['estrategia'].unique()
        colors = ['gray', 'blue', 'green', 'orange', 'red']
        
        # 1. Evolución temporal
        ax = axes[0, 0]
        for estrategia, color in zip(estrategias, colors):
            df_est = df_series[df_series['estrategia'] == estrategia]
            linestyle = '--' if estrategia == 'Sin intervención' else '-'
            linewidth = 2 if estrategia == 'Sin intervención' else 2.5
            ax.plot(df_est['t'], df_est['D'], linewidth=linewidth,
                   label=estrategia, color=color, linestyle=linestyle)
        
        ax.set_xlabel('Tiempo (t)', fontsize=12)
        ax.set_ylabel('Desinformados', fontsize=12)
        ax.set_title('Impacto de Intervenciones en Desinformación', 
                    fontweight='bold', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Reducción de prevalencia
        ax = axes[0, 1]
        baseline = df_resultados[df_resultados['estrategia'] == 'Sin intervención']['prevalencia_maxima'].values[0]
        reducciones = [(baseline - row['prevalencia_maxima'])/baseline * 100 
                      for _, row in df_resultados.iterrows()]
        
        bars = ax.barh(range(len(estrategias)), reducciones, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(estrategias)))
        ax.set_yticklabels(estrategias, fontsize=10)
        ax.set_xlabel('Reducción de Prevalencia Máxima (%)', fontsize=12)
        ax.set_title('Efectividad de Intervenciones', fontweight='bold', fontsize=13)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, red in zip(bars, reducciones):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{red:.1f}%', ha='left', va='center', fontweight='bold')
        
        # 3. Métricas comparadas
        ax = axes[1, 0]
        metricas_comp = df_resultados[['estrategia', 'tiempo_al_pico', 
                                       'duracion_epidemia', 'tasa_ataque_final']].set_index('estrategia')
        metricas_comp.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen'],
                          alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Valor', fontsize=12)
        ax.set_title('Métricas Temporales por Intervención', fontweight='bold', fontsize=13)
        ax.set_xticklabels(estrategias, rotation=45, ha='right', fontsize=9)
        ax.legend(['Tiempo al pico', 'Duración', 'Tasa de ataque'], fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Costo-efectividad (conceptual)
        ax = axes[1, 1]
        # Asignar costos relativos ficticios
        costos = [0, 3, 2, 1, 4]  # Sin intervención, fact-check, educación, moderación, combinada
        efectividades = reducciones
        
        scatter = ax.scatter(costos, efectividades, s=300, c=range(len(estrategias)),
                           cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidths=2)
        
        for i, estrategia in enumerate(estrategias):
            ax.annotate(estrategia, (costos[i], efectividades[i]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                            facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Costo Relativo de Implementación', fontsize=12)
        ax.set_ylabel('Efectividad (% Reducción Prevalencia)', fontsize=12)
        ax.set_title('Análisis Costo-Efectividad', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparacion_intervenciones.png",
                   dpi=300, bbox_inches='tight')
        print(f" Gráfica guardada: comparacion_intervenciones.png")
        plt.show()


if __name__ == "__main__":
    
    # Configuración base
    params_red_base = {
        'n_nodos': 400,
        'k_vecinos': 6,
        'p_rewire': 0.2,
        'm_hubs': 3,
        'proporcion_hubs': 0.08,
        'seed': 42
    }
    
    params_sim_base = {
        'T': 250,
        'beta0': 0.1,
        'beta1': 0.4,
        'gamma0': 0.22,
        'gamma1': 0.03,
        'delta': 0.1,
        'eta': 0.06,
        'lambd': 0.35,
        'seed_desinfo': 5,
        'no_involucrados': 0.1,
        'reinforcement': True,
        'reinforcement_rate': 0.02
    }
    
    # Crear experimento
    experimento = ExperimentoComparativo(output_dir="Experimentos")
    
    print("\n" + "="*70)
    print("SUITE DE EXPERIMENTOS COMPARATIVOS")
    print("="*70)
    
    # 1. Sensibilidad a beta0 (susceptibilidad individual)
    print("\n[1/4] Análisis de sensibilidad: beta0")
    df_beta0 = experimento.experimento_variacion_parametro(
        parametro='beta0',
        valores=[0.05, 0.075, 0.1, 0.125, 0.15],
        params_base_red=params_red_base,
        params_base_sim=params_sim_base,
        n_replicas=3
    )
    
    # 2. Sensibilidad a gamma0 (corrección individual)
    print("\n[2/4] Análisis de sensibilidad: gamma0")
    df_gamma0 = experimento.experimento_variacion_parametro(
        parametro='gamma0',
        valores=[0.15, 0.2, 0.25, 0.3, 0.35],
        params_base_red=params_red_base,
        params_base_sim=params_sim_base,
        n_replicas=3
    )
    
    # 3. Comparación de topologías
    print("\n[3/4] Comparación de topologías de red")
    df_topologias = experimento.experimento_tipos_red(
        params_sim=params_sim_base,
        n_nodos=400
    )
    
    # 4. Estrategias de intervención
    print("\n[4/4] Evaluación de estrategias de intervención")
    df_intervenciones = experimento.experimento_intervencion(
        params_base_red=params_red_base,
        params_base_sim=params_sim_base
    )
    
    print("\n" + "="*70)
    print("SUITE DE EXPERIMENTOS COMPLETADA")
    print(f"Todos los resultados guardados en: {experimento.output_dir}")
    print("="*70)