import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from math import exp

from SocialNetwork import SocialNetwork
from Agent import Agent, EstadoAgente, TipoAgente

class Simulation:
    """
    Clase que une SocialNetwork y Agent y corre la simulación SDN.
    """

    def __init__(self, network: SocialNetwork, seed: int = None):
        self.net = network
        self.seed = seed
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

    def seed_initial_states(self, n_seed_desinfo=5, prop_no_involucrados=0.05):
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
        total_activity = sum(a.actividad for a in self.agents.values())
        actives = sum(1 for a in self.agents.values() if a.esta_activo())
        if total_activity <= 0:
            return 0.0
        return actives / len(self.agents)

    def tendencia_temporal(self, t: int, t0: int = 0, lambd: float = 0.0):
        """
        T(t) para involucramiento: ejemplo simple e^{-lambda*(t-t0)}.
        Si lambd==0 devuelve 1.
        """
        if lambd <= 0:
            return 1.0
        return exp(-lambd * max(0, t - t0))

    def run(self,
            T: int = 100,
            beta0: float = 0.05,
            beta1: float = 0.3,
            gamma0: float = 0.2,
            gamma1: float = 0.01,
            delta: float = 0.05,
            eta: float = 0.03,
            lambd: float = 0.35,
            seed_desinfo: int = 5,
            no_involucrados: int = 0.1,
            reinforcement: bool = True,
            reinforcement_rate: float = 0.02,
            verbose: bool = True):
        """
        Ejecuta la simulación y retorna un DataFrame con métricas por paso.
        Parámetros: tasas que usan los agentes (beta0,beta1,gamma0,gamma1,...)
            Los parámetros 'default' fueron establecidos según bibliografía del diseño orignal
        reinforcement: si True aplica un pequeño ajuste de susceptibilidad según A_D (refuerzo).
        """
        print("\nSimulación iniciada...")

        if self.seed is not None:
            np.random.seed(self.seed)

        # Semilla inicial, estados de agentes aleatorios
        self.seed_initial_states(n_seed_desinfo=seed_desinfo, prop_no_involucrados=no_involucrados)

        logs = []
        n = len(self.agents)

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
                p_sd = agent.calcular_prob_propagacion(A_D[nodo], beta0, beta1)
                if agent.es_susceptible():
                    if np.random.random() < p_sd:
                        transiciones.append((nodo, EstadoAgente.DESINFORMADO))
                        continue  # ya pasó este paso, omitimos otras reglas

                # D -> S (corrección)
                p_ds = agent.calcular_prob_correccion(A_S[nodo], gamma0, gamma1)
                if agent.es_desinformado():
                    if np.random.random() < p_ds:
                        transiciones.append((nodo, EstadoAgente.SUSCEPTIBLE))
                        continue

                # Decaimiento a NO_INVOLUCRADO
                p_decay = agent.calcular_prob_decaimiento(viralidad, delta)
                if not agent.es_no_involucrado():
                    if np.random.random() < p_decay:
                        transiciones.append((nodo, EstadoAgente.NO_INVOLUCRADO))
                        continue

                # N -> S (re-entrada)
                p_ns = agent.calcular_prob_involucramiento(viralidad, tendencia, eta)
                if agent.es_no_involucrado():
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
                    # regla simple: s <- s + reinforcement_rate * exposure * (1 - s)
                    new_s = agent.susceptibilidad + reinforcement_rate * exposure * (1 - agent.susceptibilidad)
                    agent.susceptibilidad = min(1.0, max(0.0, new_s))

            # 5) Métricas y logging
            counts = {
                't': t,
                'S': sum(1 for a in self.agents.values() if a.es_susceptible()),
                'D': sum(1 for a in self.agents.values() if a.es_desinformado()),
                'N': sum(1 for a in self.agents.values() if a.es_no_involucrado()),
                'viralidad': viralidad,
                'avg_susceptibilidad': np.mean([a.susceptibilidad for a in self.agents.values()]),
                'avg_actividad': np.mean([a.actividad for a in self.agents.values()])
            }
            logs.append(counts)

            if verbose and (t % max(1, T//10) == 0 or t == T-1):
                print(f"t={t:03d}  S={counts['S']}  D={counts['D']}  N={counts['N']}  viral={viralidad:.3f}")

        df = pd.DataFrame(logs)
        return df

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # Crear carpeta de resultados si no existe
    os.makedirs("SimResults", exist_ok=True)

    # Parámetros base según tu tabla
    params = {
        "beta0": 0.1,   # Propagación a nivel de agente
        "beta1": 0.4,   # Propagación a nivel de red
        "gamma0": 0.22, # Corrección a nivel de agente
        "gamma1": 0.03, # Corrección a nivel de red
        "delta": 0.1,   # Decaimiento de atención
        "eta": 0.05,    # Reingreso desde No Involucrado
        "lambd": 0.35, # Decaimiento de tendencia
    }

    # Crear red híbrida con 200 nodos
    red = SocialNetwork(
        n_nodos=400,
        k_vecinos=6,           # Cada nodo conectado a 6 vecinos cercanos (small-world)
        p_rewire=0.2,          # 20% de reconexiones (shortcuts entre comunidades)
        m_hubs=3,              # Cada usuario regular sigue a ~3 influencers
        proporcion_hubs=0.08,  # 8% de la población son hubs/influencers
        seed=42
    )

    # Crear simulación
    sim = Simulation(
        network=red,
        seed=42
    )

    # Ejecutar simulación
    df = sim.run(
        T=50,
        **params,
        seed_desinfo=23,
        no_involucrados=0.05,
        reinforcement=True,
        reinforcement_rate=0.01
    )

    # Guardar resultados
    run_id = len(os.listdir("SimResults")) + 1
    filename = f"SimResults/run{run_id}.csv"
    df.to_csv(filename, index=False)
    print(f"Resultados guardados en {filename}")

    # Mostrar métricas finales
    final = df.iloc[-1]
    print("\n--- Métricas Finales ---")
    print(f"Susceptibles: {final['S']}")
    print(f"Desinformados: {final['D']}")
    print(f"No involucrados: {final['N']}")
    print(f"Viralidad final: {final['viralidad']:.3f}")
    print(f"Susceptibilidad media: {final['avg_susceptibilidad']:.3f}")

    # Gráfico de evolución temporal
    plt.figure(figsize=(8,5))
    plt.plot(df["t"], df["S"], label="Susceptibles (S)")
    plt.plot(df["t"], df["D"], label="Desinformados (D)")
    plt.plot(df["t"], df["N"], label="No involucrados (N)")
    plt.xlabel("Tiempo (t)")
    plt.ylabel("Número de agentes")
    plt.title("Evolución de estados en el modelo SDN")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
