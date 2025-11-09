import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from math import exp
import os

from SocialNetwork import SocialNetwork
from Agent import Agent, EstadoAgente, TipoAgente

class Simulation:
    """
    Clase que une SocialNetwork y Agent y corre la simulación SDN.
    Incluye recolección de métricas detalladas y trazabilidad completa.
    """

    def __init__(self, network: SocialNetwork, seed: int = None, track_history: bool = True):
        """
        Args:
            network: Red social
            seed: Semilla para reproducibilidad
            track_history: Si True, guarda historia completa de estados de agentes
        """
        self.net = network
        self.seed = seed
        self.track_history = track_history
        if seed is not None:
            np.random.seed(seed)

        # Crear agentes y mapear a nodos del grafo
        self.agents: Dict[int, Agent] = {}
        for n in self.net.G.nodes():
            a = Agent(id_agente=int(n), seed=seed)  # deja que Agent inicialice tipo/atributos
            # si el nodo fue marcado hub/super_propagador en la red, actualizar
            if self.net.G.nodes[n].get('es_hub', False):
                a.marcar_como_hub()
            if self.net.G.nodes[n].get('super_propagador', False):
                a.marcar_como_super_propagador()
            self.agents[n] = a
        # Historia de estados (si está activado)
        self.history = [] if track_history else None
        
        # Métricas adicionales por paso
        self.metricas_detalladas = []

    def seed_initial_states(self, n_seed_desinfo=5, prop_no_involucrados=0.05):
        """Inicializa estados de agentes."""
        # marcas para desinformados
        nodes = list(self.net.G.nodes())
        seeds = list(np.random.choice(nodes, size=n_seed_desinfo, replace=False))
        for s in seeds:
            self.agents[s].cambiar_estado(EstadoAgente.DESINFORMADO)

        # marcar No Involucrados
        n_no = int(len(nodes) * prop_no_involucrados)
        remaining = [n for n in nodes if n not in seeds]
        if n_no > 0:
            no_nodes = list(np.random.choice(remaining, size=n_no, replace=False))
            for nn in no_nodes:
                self.agents[nn].cambiar_estado(EstadoAgente.NO_INVOLUCRADO)

    def _compute_A_vectors(self):
        """
        Construye dicts que la red puede usar en calcular_suma_ponderada_vecinos:
          - atributo_vecinos_desinfo: {node: actividad} solo para vecinos que están DESINFORMADOS
          - atributo_vecinos_susceptible: {node: actividad} solo para vecinos SUSCEPTIBLES
        Construye vectores A_D y A_S para cálculos de probabilidad.
        """
        atributo_desinfo = {}
        atributo_sus = {}
        for nid, agent in self.agents.items():
            act = agent.obtener_actividad_efectiva()
            if agent.es_desinformado():
                atributo_desinfo[nid] = act
            if agent.es_susceptible():
                atributo_sus[nid] = act
        return atributo_desinfo, atributo_sus

    def compute_viralidad_by_activity(self):
        """Calcula viralidad como proporción de agentes activos."""
        actives = sum(1 for a in self.agents.values() if a.esta_activo())
        return actives / len(self.agents)

    def tendencia_temporal(self, t: int, t0: int = 0, lambd: float = 0.0):
        """
        T(t) para involucramiento: ejemplo simple e^{-lambda*(t-t0)}.
        Si lambd==0 devuelve 1.
        """
        if lambd <= 0:
            return 1.0
        return exp(-lambd * max(0, t - t0))
    
    def _guardar_snapshot(self, t: int):
        """Guarda snapshot del estado actual de todos los agentes."""
        if self.history is not None:
            snapshot = {
                't': t,
                'estados': {nid: agent.estado.value for nid, agent in self.agents.items()},
                'susceptibilidades': {nid: agent.susceptibilidad for nid, agent in self.agents.items()}
            }
            self.history.append(snapshot)
    
    def _calcular_metricas_paso(self, t: int, viralidad: float, A_D: Dict, A_S: Dict) -> Dict:
        """Calcula métricas detalladas para el paso actual."""
        # Conteos por estado
        n_S = sum(1 for a in self.agents.values() if a.es_susceptible())
        n_D = sum(1 for a in self.agents.values() if a.es_desinformado())
        n_N = sum(1 for a in self.agents.values() if a.es_no_involucrado())
        
        # Métricas por tipo de agente
        tipos_D = {'credulo': 0, 'esceptico': 0, 'verificador': 0}
        for a in self.agents.values():
            if a.es_desinformado():
                tipos_D[a.tipo.value] += 1
        
        # Súper propagadores desinformados
        n_super_props_D = sum(1 for a in self.agents.values() 
                              if a.es_desinformado() and a.es_super_propagador)
        n_hubs_D = sum(1 for a in self.agents.values() 
                       if a.es_desinformado() and a.es_hub)
        
        # Exposición promedio a desinformación
        exposiciones = [A_D[nid] for nid in self.agents.keys()]
        exposicion_media = np.mean(exposiciones) if exposiciones else 0
        exposicion_max = np.max(exposiciones) if exposiciones else 0
        
        # Actividad total de desinformados
        actividad_D = sum(a.actividad for a in self.agents.values() if a.es_desinformado())
        
        metricas = {
            't': t,
            'S': n_S,
            'D': n_D,
            'N': n_N,
            'viralidad': viralidad,
            'avg_susceptibilidad': np.mean([a.susceptibilidad for a in self.agents.values()]),
            'avg_actividad': np.mean([a.actividad for a in self.agents.values()]),
            'D_credulos': tipos_D['credulo'],
            'D_escepticos': tipos_D['esceptico'],
            'D_verificadores': tipos_D['verificador'],
            'D_super_propagadores': n_super_props_D,
            'D_hubs': n_hubs_D,
            'exposicion_media_desinfo': exposicion_media,
            'exposicion_max_desinfo': exposicion_max,
            'actividad_total_desinfo': actividad_D
        }
        
        return metricas

    def run(self,
            T: int = 100,
            beta0: float = 0.05,
            beta1: float = 0.3,
            gamma0: float = 0.2,
            gamma1: float = 0.01,
            delta: float = 0.05,
            eta: float = 0.1,
            lambd: float = 0.35,
            seed_desinfo: int = 5,
            no_involucrados: float = 0.1,
            reinforcement: bool = True,
            reinforcement_rate: float = 0.02,
            verbose: bool = True,
            save_snapshots: bool = False) -> pd.DataFrame:
        """
        Ejecuta la simulación y retorna un DataFrame con métricas por paso.
        Parámetros: tasas que usan los agentes (beta0,beta1,gamma0,gamma1,...)
            Los parámetros 'default' fueron establecidos según bibliografía del diseño orignal
        reinforcement: si True aplica un pequeño ajuste de susceptibilidad según A_D (refuerzo).
        Ejecuta la simulación con recolección detallada de métricas.
        
        Args:
            T: Número de pasos temporales
            beta0, beta1: Tasas de propagación
            gamma0, gamma1: Tasas de corrección
            delta: Tasa de decaimiento
            eta: Tasa de involucramiento
            lambd: Decaimiento de tendencia
            seed_desinfo: Número inicial de desinformados
            no_involucrados: Proporción inicial de no involucrados
            reinforcement: Si aplicar refuerzo de susceptibilidad
            reinforcement_rate: Tasa de refuerzo
            verbose: Si imprimir progreso
            save_snapshots: Si guardar snapshots de estados
            
        Returns:
            DataFrame con métricas por paso temporal
        """
        if verbose:
            print("\n" + "="*70)
            print("SIMULACIÓN SDN INICIADA")
            print("="*70)
            print(f"Parámetros:")
            print(f"  T={T}, β0={beta0}, β1={beta1}, γ0={gamma0}, γ1={gamma1}")
            print(f"  δ={delta}, η={eta}, λ={lambd}")
            print(f"  Seeds desinformación: {seed_desinfo}")
            print(f"  Proporción no involucrados: {no_involucrados}")
            print(f"  Reinforcement: {reinforcement}")
            print("="*70 + "\n")

        if self.seed is not None:
            np.random.seed(self.seed)

        # Inicializar estados
        # Semilla inicial, estados de agentes aleatorios
        self.seed_initial_states(n_seed_desinfo=seed_desinfo, prop_no_involucrados=no_involucrados)
        
        # Guardar parámetros para referencia
        self.parametros_simulacion = {
            'T': T, 'beta0': beta0, 'beta1': beta1,
            'gamma0': gamma0, 'gamma1': gamma1,
            'delta': delta, 'eta': eta, 'lambd': lambd,
            'seed_desinfo': seed_desinfo,
            'no_involucrados': no_involucrados,
            'reinforcement': reinforcement,
            'reinforcement_rate': reinforcement_rate
        }

        logs = []
        intervalo_reporte = max(1, T // 10)

        for t in range(T):
            # 1) Construir vectores A_D y A_S
            atributo_desinfo, atributo_sus = self._compute_A_vectors()

            A_D = {}
            A_S = {}
            for nodo in self.agents.keys():
                A_D[nodo] = self.net.calcular_suma_ponderada_vecinos(nodo, atributo_desinfo)
                A_S[nodo] = self.net.calcular_suma_ponderada_vecinos(nodo, atributo_sus)

            # 2) viralidad y tendencia
            viralidad = self.compute_viralidad_by_activity()
            tendencia = self.tendencia_temporal(t, t0=0, lambd=lambd)

            # 3) Iterar agentes y aplicar reglas S->D, D->S, decaimiento, N->S
            # Hacemos en dos pasos: calcular probabilidades y aplicar transiciones
            # para evitar dependencias ordenadas (determinismo de paso simultáneo)
            transiciones = []  # (nodo, nuevo_estado)

            for nodo, agent in self.agents.items():
                # S -> D
                if agent.es_susceptible():
                    p_sd = agent.calcular_prob_propagacion(A_D[nodo], beta0, beta1)
                    if np.random.random() < p_sd:
                        transiciones.append((nodo, EstadoAgente.DESINFORMADO))
                        continue

                # D -> S (corrección)
                if agent.es_desinformado():
                    p_ds = agent.calcular_prob_correccion(A_S[nodo], gamma0, gamma1)
                    if np.random.random() < p_ds:
                        transiciones.append((nodo, EstadoAgente.SUSCEPTIBLE))
                        continue

                # Decaimiento a NO_INVOLUCRADO [S,D] -> N
                if not agent.es_no_involucrado():
                    p_decay = agent.calcular_prob_decaimiento(viralidad, delta)
                    if np.random.random() < p_decay:
                        transiciones.append((nodo, EstadoAgente.NO_INVOLUCRADO))
                        continue

                # N -> S (re-entrada)
                if agent.es_no_involucrado():
                    p_ns = agent.calcular_prob_involucramiento(viralidad, tendencia, eta)
                    if np.random.random() < p_ns:
                        transiciones.append((nodo, EstadoAgente.SUSCEPTIBLE))
                        continue

            # Aplicar transiciones
            for nodo, nuevo_estado in transiciones:
                self.agents[nodo].cambiar_estado(nuevo_estado)

            # 4) Reinforcement opcional: ajustar susceptibilidad por exposición acumulada A_D
            if reinforcement:
                for nodo, agent in self.agents.items():
                    # sólo ajustar si recibió exposición relevante
                    exposure = A_D[nodo]
                    new_s = agent.susceptibilidad + reinforcement_rate * exposure * (1 - agent.susceptibilidad)
                    agent.susceptibilidad = min(1.0, max(0.0, new_s))

            # 5) Guardar snapshot
            if save_snapshots:
                self._guardar_snapshot(t)

            # 6) Calcular y guardar métricas
            metricas = self._calcular_metricas_paso(t, viralidad, A_D, A_S)
            logs.append(metricas)

            # 7) Reporte de progreso
            if verbose and (t % intervalo_reporte == 0 or t == T-1):
                print(f"t={t:03d}  S={metricas['S']:4d}  D={metricas['D']:4d}  "
                      f"N={metricas['N']:4d}  V={viralidad:.3f}  "
                      f"Exp={metricas['exposicion_media_desinfo']:.3f}")

        df = pd.DataFrame(logs)
        
        if verbose:
            print("\n" + "="*70)
            print("SIMULACIÓN COMPLETADA")
            print("="*70)
        
        return df

    def get_estado_agentes(self) -> pd.DataFrame:
        """
        Retorna DataFrame con estado actual de todos los agentes.
        Útil para análisis post-simulación.
        """
        data = []
        for nid, agent in self.agents.items():
            info = agent.obtener_info()
            info['nodo_id'] = nid
            data.append(info)
        
        return pd.DataFrame(data)
    
    def exportar_historia(self, filename: str):
        """Exporta historia completa de estados a archivo."""
        if self.history is None:
            print("No hay historia guardada. Activar track_history=True")
            return
        
        import json
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Historia exportada a {filename}")


def ejecutar_simulacion_completa(params_red: Dict = None,
                                 params_sim: Dict = None,
                                 output_dir: str = "AnalysisResults",
                                 run_id: str = None) -> tuple:
    """
    Función wrapper que ejecuta simulación completa con análisis integrado.
    
    Args:
        params_red: Parámetros para crear la red
        params_sim: Parámetros para la simulación
        output_dir: Directorio de salida
        run_id: Identificador de la corrida
        
    Returns:
        (simulation, dataframe, analyzer, reporte)
    """
    # Importar analyzer aquí para evitar dependencia circular
    from Analyzer import SimulationAnalyzer
    
    # Parámetros por defecto
    if params_red is None:
        params_red = {
            'n_nodos': 400,
            'k_vecinos': 6,
            'p_rewire': 0.2,
            'm_hubs': 3,
            'proporcion_hubs': 0.08,
            'seed': 42
        }
    
    if params_sim is None:
        params_sim = {
            'T': 100,
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
    
    # Crear identificador único si no se proporciona
    if run_id is None:
        from datetime import datetime
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*70}")
    print(f"EJECUCIÓN COMPLETA - Run ID: {run_id}")
    print(f"{'='*70}\n")
    
    # 1. Crear red
    print("PASO 1: Creando red social...")
    print("-" * 70)
    red = SocialNetwork(**params_red)
    red.asignar_super_propagadores()
    
    # 2. Crear simulación
    print("\nPASO 2: Inicializando simulación...")
    print("-" * 70)
    sim = Simulation(network=red, seed=params_red.get('seed', 42), 
                            track_history=True)
    
    # 3. Ejecutar simulación
    print("\nPASO 3: Ejecutando simulación...")
    print("-" * 70)
    df = sim.run(**params_sim, verbose=True, save_snapshots=False)
    
    # 4. Guardar resultados intermedios
    print("\nPASO 4: Guardando resultados intermedios...")
    print("-" * 70)
    results_dir = f"{output_dir}/Run_{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    df.to_csv(f"{results_dir}/timeseries.csv", index=False)
    sim.get_estado_agentes().to_csv(f"{results_dir}/estado_agentes_final.csv", index=False)
    print(f" Resultados guardados en {results_dir}")
    
    # 5. Análisis completo
    print("\nPASO 5: Ejecutando análisis completo...")
    print("-" * 70)
    analyzer = SimulationAnalyzer(output_dir=results_dir)
    reporte = analyzer.generar_reporte_completo(sim, df, red, save=True)
    
    # 6. Guardar parámetros
    import json
    parametros_completos = {
        'run_id': run_id,
        'parametros_red': params_red,
        'parametros_simulacion': params_sim
    }
    with open(f"{results_dir}/parametros.json", 'w') as f:
        json.dump(parametros_completos, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"EJECUCIÓN COMPLETA FINALIZADA")
    print(f"Todos los resultados en: {results_dir}")
    print(f"{'='*70}\n")
    
    return sim, df, analyzer, reporte


# EJEMPLO DE USO
if __name__ == "__main__":
    # Configuración de red
    params_red = {
        'n_nodos': 400,
        'k_vecinos': 6,
        'p_rewire': 0.2,
        'm_hubs': 3,
        'proporcion_hubs': 0.08,
        'seed': 42
    }
    
    # Configuración de simulación (parámetros calibrados)
    params_sim = {
        'T': 250,
        'beta0': 0.1,    # Propagación a nivel de agente
        'beta1': 0.4,    # Propagación a nivel de red
        'gamma0': 0.22,  # Corrección a nivel de agente
        'gamma1': 0.03,  # Corrección a nivel de red
        'delta': 0.1,    # Decaimiento de atención
        'eta': 0.06,     # Reingreso desde No Involucrado (Involucramiento)
        'lambd': 0.35,   # Decaimiento de tendencia
        'seed_desinfo': 5,
        'no_involucrados': 0.1,
        'reinforcement': True,
        'reinforcement_rate': 0.02
    }
    
    # Ejecutar simulación completa con análisis
    sim, df, analyzer, reporte = ejecutar_simulacion_completa(
        params_red=params_red,
        params_sim=params_sim,
        output_dir="AnalysisResults",
        run_id="baseline"
    )