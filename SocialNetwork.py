import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class SocialNetwork:
    """
    Clase para representar la estructura de red social para simulación de desinformación.
    Implementa topología híbrida: Small-world (comunidades) + Scale-free (influencers).
    """
    
    def __init__(self, n_nodos: int, k_vecinos: int = 6, p_rewire: float = 0.3,
                 m_hubs: int = 2, proporcion_hubs: float = 0.10, seed: int = None):
        """
        Inicializa la red social con topología híbrida.
        
        Args:
            n_nodos: Número total de usuarios en la red
            k_vecinos: Número de vecinos cercanos para small-world (clustering local)
            p_rewire: Probabilidad de reconexión para crear shortcuts (0.1-0.3 recomendado)
            m_hubs: Número de conexiones que se añaden a cada hub
            proporcion_hubs: Proporción de nodos que serán hubs/influencers (0.05-0.15)
            seed: Semilla para reproducibilidad
        """
        self.n_nodos = n_nodos
        self.k_vecinos = k_vecinos
        self.p_rewire = p_rewire
        self.m_hubs = m_hubs
        self.proporcion_hubs = proporcion_hubs
        self.seed = seed
        
        if seed:
            np.random.seed(seed)
        
        # Crear topología híbrida
        print("Creando topología híbrida (Small-world + Scale-free)...")
        self.G = self._crear_topologia_hibrida()
        
        # Inicializar pesos
        self._inicializar_pesos()
        
        # Identificar súper propagadores (serán los hubs creados)
        self.super_propagadores = []
        
    def _crear_topologia_hibrida(self) -> nx.DiGraph:
        """
        Crea topología híbrida combinando Small-world y Scale-free.
        
        Proceso:
        1. Crear base Small-world (comunidades locales con shortcuts)
        2. Identificar nodos que serán hubs/influencers
        3. Añadir conexiones preferencial attachment a esos hubs (scale-free)
        """
        # PASO 1: Base Small-world (comunidades locales)
        print(f"  - Generando base small-world ({self.n_nodos} nodos, k={self.k_vecinos})...")
        G_base = nx.watts_strogatz_graph(
            n=self.n_nodos, 
            k=self.k_vecinos, 
            p=self.p_rewire, 
            seed=self.seed
        )
        
        # Convertir a dirigido
        G = G_base.to_directed()
        
        # PASO 2: Seleccionar nodos que serán hubs/influencers
        n_hubs = max(1, int(self.n_nodos * self.proporcion_hubs))
        print(f"  - Seleccionando {n_hubs} nodos como hubs/influencers...")
        
        # Seleccionar aleatoriamente o por grado actual
        # Opción: Seleccionar los que ya tienen más conexiones (preferencia inicial)
        grados = dict(G.degree())
        nodos_ordenados = sorted(grados.items(), key=lambda x: x[1], reverse=True)
        hubs = [nodo for nodo, _ in nodos_ordenados[:n_hubs]]
        
        # Marcar como hubs
        for hub in hubs:
            G.nodes[hub]['es_hub'] = True
        
        # PASO 3: Añadir conexiones scale-free (preferential attachment)
        print(f"  - Añadiendo conexiones preferential attachment a hubs...")
        self._añadir_conexiones_scale_free(G, hubs)
        
        print(f"  Red híbrida creada: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
        
        return G
    
    def _añadir_conexiones_scale_free(self, G: nx.DiGraph, hubs: List[int]):
        """
        Añade conexiones usando preferential attachment hacia los hubs.
        Simula que usuarios "siguen" a influencers.
        """
        # Para cada nodo regular, tiene probabilidad de conectarse a hubs
        nodos_regulares = [n for n in G.nodes() if n not in hubs]
        
        for nodo in nodos_regulares:
            # Calcular probabilidades de conexión basadas en grado actual de hubs
            grados_hubs = np.array([G.in_degree(hub) + 1 for hub in hubs])
            probabilidades = grados_hubs / grados_hubs.sum()
            
            # Seleccionar m_hubs hubs para seguir (sin reemplazo)
            n_conexiones = min(self.m_hubs, len(hubs))
            hubs_seleccionados = np.random.choice(
                hubs, 
                size=n_conexiones, 
                replace=False,
                p=probabilidades
            )
            
            # Añadir aristas: nodo regular -> hub (el regular "sigue" al hub)
            for hub in hubs_seleccionados:
                if not G.has_edge(nodo, hub):
                    G.add_edge(nodo, hub)
        
        # También hacer que algunos hubs se sigan entre sí
        for i, hub1 in enumerate(hubs):
            for hub2 in hubs[i+1:]:
                if np.random.random() < 0.3:  # 30% probabilidad
                    G.add_edge(hub1, hub2)
                if np.random.random() < 0.3:
                    G.add_edge(hub2, hub1)
    
    def _inicializar_pesos(self):
        """
        Inicializa pesos unidireccionales en las aristas.
        Los pesos representan el nivel de confianza entre usuarios.
        Los hubs reciben mayor confianza (pesos más altos).
        Normaliza para que la suma de pesos salientes de cada nodo sea 1.
        """
        # Asignar pesos con sesgo hacia hubs
        for u, v in self.G.edges():
            # Si v es un hub, el peso base es mayor (más confianza)
            if self.G.nodes[v].get('es_hub', False):
                peso_base = np.random.uniform(0.5, 1.0)  # Mayor confianza a hubs
            else:
                peso_base = np.random.uniform(0.1, 0.6)  # Confianza normal
            
            self.G[u][v]['weight'] = peso_base
        
        # Normalizar pesos para cada nodo
        self._normalizar_pesos()
    
    def _normalizar_pesos(self):
        """
        Normaliza los pesos salientes de cada nodo para que sumen 1.
        Cumple con: Σ w_ij = 1 para cada nodo i sobre sus vecinos j.
        """
        for nodo in self.G.nodes():
            # Obtener vecinos salientes (a quienes sigue/conecta)
            vecinos_salientes = list(self.G.successors(nodo))
            
            if len(vecinos_salientes) > 0:
                # Sumar pesos actuales
                suma_pesos = sum(self.G[nodo][vecino]['weight'] 
                               for vecino in vecinos_salientes)
                
                # Normalizar
                if suma_pesos > 0:
                    for vecino in vecinos_salientes:
                        self.G[nodo][vecino]['weight'] /= suma_pesos
    
    def asignar_super_propagadores(self, proporcion: float = None, 
                                   criterio: str = 'grado'):
        """
        Identifica y marca súper propagadores en la red.
        Por defecto, los hubs ya son súper propagadores.
        
        Args:
            proporcion: Proporción adicional de nodos (si None, usa solo hubs)
            criterio: 'grado' (por número de conexiones) o 'betweenness' (por centralidad)
        
        Returns:
            Lista de IDs de súper propagadores
        """
        # Los hubs son automáticamente súper propagadores
        self.super_propagadores = [n for n in self.G.nodes() 
                                  if self.G.nodes[n].get('es_hub', False)]
        
        # Si se especifica proporción adicional, añadir más
        if proporcion is not None:
            n_super_total = max(len(self.super_propagadores), 
                               int(self.n_nodos * proporcion))
            n_adicionales = n_super_total - len(self.super_propagadores)
            
            if n_adicionales > 0:
                # Nodos que no son hubs
                no_hubs = [n for n in self.G.nodes() 
                          if not self.G.nodes[n].get('es_hub', False)]
                
                if criterio == 'grado':
                    grados = {n: self.G.degree(n) for n in no_hubs}
                    ordenados = sorted(grados.items(), key=lambda x: x[1], reverse=True)
                elif criterio == 'betweenness':
                    centralidad = nx.betweenness_centrality(self.G)
                    ordenados = sorted([(n, centralidad[n]) for n in no_hubs], 
                                     key=lambda x: x[1], reverse=True)
                
                adicionales = [nodo for nodo, _ in ordenados[:n_adicionales]]
                self.super_propagadores.extend(adicionales)
        
        # Marcar en el grafo
        for nodo in self.super_propagadores:
            self.G.nodes[nodo]['super_propagador'] = True
        
        print(f"{len(self.super_propagadores)} súper propagadores identificados")
        return self.super_propagadores
    
    def obtener_matriz_adyacencia(self) -> np.ndarray:
        """
        Retorna la matriz de adyacencia con pesos.
        
        Returns:
            Matriz numpy de tamaño (n_nodos x n_nodos)
        """
        return nx.to_numpy_array(self.G, weight='weight')
    
    def obtener_vecinos_entrantes(self, nodo: int) -> List[int]:
        """Retorna lista de vecinos que apuntan hacia el nodo (influencias)."""
        return list(self.G.predecessors(nodo))
    
    def obtener_vecinos_salientes(self, nodo: int) -> List[int]:
        """Retorna lista de vecinos hacia los que apunta el nodo (a quienes influye)."""
        return list(self.G.successors(nodo))
    
    def obtener_peso(self, nodo_origen: int, nodo_destino: int) -> float:
        """Retorna el peso de confianza de origen hacia destino."""
        if self.G.has_edge(nodo_origen, nodo_destino):
            return self.G[nodo_origen][nodo_destino]['weight']
        return 0.0
    
    def calcular_suma_ponderada_vecinos(self, nodo: int, 
                                       atributo_vecinos: Dict[int, float]) -> float:
        """
        Calcula A_x(i) = Σ a_j * w_ji para vecinos entrantes del nodo.
        Esta es la función clave para el modelo SDN del documento.
        
        Args:
            nodo: ID del nodo objetivo
            atributo_vecinos: Diccionario {id_vecino: valor_atributo}
                             (por ejemplo, actividad de vecinos desinformados)
        
        Returns:
            Suma ponderada de atributos de vecinos entrantes
        """
        vecinos_entrantes = self.obtener_vecinos_entrantes(nodo)
        suma = 0.0
        
        for vecino in vecinos_entrantes:
            if vecino in atributo_vecinos:
                peso = self.obtener_peso(vecino, nodo)
                suma += atributo_vecinos[vecino] * peso
        
        return suma
    
    def identificar_comunidades(self, metodo: str = 'louvain') -> Dict[int, int]:
        """
        Identifica comunidades en la red.
        Args:
            metodo: 'louvain', 'greedy' o 'label'
        Returns:
            Diccionario {nodo: id_comunidad}
        """
        G_undir = self.G.to_undirected()
        
        if metodo == 'louvain':
            comunidades = nx.community.louvain_communities(G_undir, seed=self.seed)
        elif metodo == 'greedy':
            comunidades = nx.community.greedy_modularity_communities(G_undir)
        else:
            comunidades = nx.community.label_propagation_communities(G_undir)
        
        partition = {}
        for i, comunidad in enumerate(comunidades):
            for nodo in comunidad:
                partition[nodo] = i
        return partition

    
    def estadisticas_red(self) -> Dict:
        """Retorna estadísticas descriptivas de la red."""
        grados_in = [d for n, d in self.G.in_degree()]
        grados_out = [d for n, d in self.G.out_degree()]
        
        stats = {
            'n_nodos': self.G.number_of_nodes(),
            'n_aristas': self.G.number_of_edges(),
            'densidad': nx.density(self.G),
            'grado_in_promedio': np.mean(grados_in),
            'grado_out_promedio': np.mean(grados_out),
            'grado_in_max': np.max(grados_in),
            'grado_out_max': np.max(grados_out),
            'coeficiente_clustering': nx.average_clustering(self.G.to_undirected()),
            'n_super_propagadores': len(self.super_propagadores),
            'n_hubs': sum(1 for n in self.G.nodes() if self.G.nodes[n].get('es_hub', False))
        }
        
        # Camino promedio
        if self.n_nodos < 1000:
            try:
                G_undir = self.G.to_undirected()
                if nx.is_connected(G_undir):
                    stats['camino_promedio'] = nx.average_shortest_path_length(G_undir)
                else:
                    # Si hay componentes desconectadas, calcular para la mayor
                    largest_cc = max(nx.connected_components(G_undir), key=len)
                    subG = G_undir.subgraph(largest_cc)
                    stats['camino_promedio'] = nx.average_shortest_path_length(subG)
                    stats['nota'] = 'Camino promedio calculado en componente principal'
            except:
                stats['camino_promedio'] = 'No calculable'
        
        return stats
    
    def visualizar_red(self, mostrar_comunidades: bool = False,
                      figsize: Tuple[int, int] = (14, 10)):
        """
        Visualiza la red con networkx.
        
        Args:
            mostrar_comunidades: Si colorear por comunidades detectadas
            figsize: Tamaño de la figura
        """
        plt.figure(figsize=figsize)
        
        # Layout para posicionar nodos
        if self.n_nodos < 200:
            pos = nx.spring_layout(self.G, seed=self.seed, k=0.5, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(self.G)
        
        # Determinar colores
        if mostrar_comunidades:
            try:
                comunidades = self.identificar_comunidades(metodo='louvain')
                n_comunidades = max(comunidades.values()) + 1
                cmap = plt.cm.get_cmap('tab20', n_comunidades)
                colores = [cmap(comunidades[n]) for n in self.G.nodes()]
            except:
                colores = 'lightblue'
        else:
            # Colorear por tipo: hubs en rojo, regulares en azul
            colores = ['red' if self.G.nodes[n].get('es_hub', False) 
                      else 'lightblue' for n in self.G.nodes()]
        
        # Tamaños proporcionales al grado
        tamaños = [50 + 10 * self.G.degree(n) for n in self.G.nodes()]
        
        # Dibujar
        nx.draw_networkx_nodes(self.G, pos, node_color=colores, 
                              node_size=tamaños, alpha=0.7)
        nx.draw_networkx_edges(self.G, pos, alpha=0.15, arrows=True, 
                              arrowsize=8, width=0.3, 
                              connectionstyle='arc3,rad=0.1')
        
        # Labels solo para redes pequeñas
        if self.n_nodos < 50:
            nx.draw_networkx_labels(self.G, pos, font_size=8)
        
        # Destacar hubs con borde
        hubs = [n for n in self.G.nodes() if self.G.nodes[n].get('es_hub', False)]
        if hubs:
            hub_pos = {n: pos[n] for n in hubs}
            hub_sizes = [50 + 10 * self.G.degree(n) for n in hubs]
            nx.draw_networkx_nodes(self.G, hub_pos, nodelist=hubs,
                                  node_color='none', edgecolors='darkred',
                                  node_size=hub_sizes, linewidths=3)
        
        plt.title(f'Red Social Híbrida (Small-world + Scale-free)\n'
                 f'{self.n_nodos} nodos, {len(hubs)} hubs/influencers (rojo)',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualizar_distribucion_grados(self):
        """Visualiza la distribución de grados (in y out)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        grados_in = [d for n, d in self.G.in_degree()]
        grados_out = [d for n, d in self.G.out_degree()]
        
        # Grado entrante (followers/influencias)
        axes[0].hist(grados_in, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Grado Entrante (In-degree)', fontsize=12)
        axes[0].set_ylabel('Frecuencia', fontsize=12)
        axes[0].set_title('Distribución de Grados Entrantes\n(Cuántos siguen a cada usuario)', 
                         fontsize=12, fontweight='bold')
        axes[0].axvline(np.mean(grados_in), color='red', linestyle='--', 
                       label=f'Media: {np.mean(grados_in):.1f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Grado saliente (following/a quién influye)
        axes[1].hist(grados_out, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Grado Saliente (Out-degree)', fontsize=12)
        axes[1].set_ylabel('Frecuencia', fontsize=12)
        axes[1].set_title('Distribución de Grados Salientes\n(A cuántos sigue cada usuario)', 
                         fontsize=12, fontweight='bold')
        axes[1].axvline(np.mean(grados_out), color='red', linestyle='--',
                       label=f'Media: {np.mean(grados_out):.1f}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def guardar_red(self, filename: str = 'red_social.gexf'):
        """Guarda la red en formato GEXF para análisis externo."""
        nx.write_gexf(self.G, filename)
        print(f" Red guardada en {filename}")
        
# EJEMPLO DE USO
if __name__ == "__main__":
    
    # Crear red híbrida con 200 nodos
    red = SocialNetwork(
        n_nodos=400,
        k_vecinos=6,           # Cada nodo conectado a 6 vecinos cercanos (small-world)
        p_rewire=0.2,          # 20% de reconexiones (shortcuts entre comunidades)
        m_hubs=3,              # Cada usuario regular sigue a ~3 influencers
        proporcion_hubs=0.08,  # 8% de la población son hubs/influencers
        seed=42
    )
    
    print("   " + "-" * 70)
    
    # Asignar súper propagadores 
    super_prop = red.asignar_super_propagadores()
    
    print("   " + "-" * 70)
    
    # Mostrar estadísticas
    print("\nEstadísticas de la red:")
    stats = red.estadisticas_red()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key:.<30} {value:.4f}")
        else:
            print(f"   {key:.<30} {value}")
            
    print("   " + "-" * 70)
    
    # Ejemplo de operaciones para la simulación
    nodo_ejemplo = 0
    vecinos_in = red.obtener_vecinos_entrantes(nodo_ejemplo)
    vecinos_out = red.obtener_vecinos_salientes(nodo_ejemplo)
    print(f"   Nodo {nodo_ejemplo}:")
    print(f"   - Recibe influencia de: {len(vecinos_in)} nodos")
    print(f"   - Influye a: {len(vecinos_out)} nodos")
    
    if vecinos_out:
        peso_ejemplo = red.obtener_peso(nodo_ejemplo, vecinos_out[0])
        print(f"   - Confianza hacia nodo {vecinos_out[0]}: {peso_ejemplo:.4f}")
        
    print("   " + "-" * 70)
    
    # Simular cálculo de A_D(i) para el modelo SDN
    # Suponer que algunos vecinos están desinformados con actividad
    actividades_ejemplo = {v: np.random.uniform(0.1, 0.9) for v in vecinos_in[:3]}
    suma_ponderada = red.calcular_suma_ponderada_vecinos(nodo_ejemplo, actividades_ejemplo)
    print(f"   A_D({nodo_ejemplo}) = {suma_ponderada:.4f}")
    
    print("   " + "-" * 70)
    
    # Verificar normalización
    if vecinos_out:
        suma_pesos = sum(red.obtener_peso(nodo_ejemplo, v) for v in vecinos_out)
        print(f"   Suma de pesos salientes del nodo {nodo_ejemplo}: {suma_pesos:.6f}")
        if abs(suma_pesos - 1.0) < 1e-6:
            print("   Normalización correcta")
        else:
            print("   Error en normalización")
    
    print("   " + "-" * 70)
    
    # Análisis de comunidades
    try:
        comunidades = red.identificar_comunidades(metodo='louvain')
        n_comunidades = len(set(comunidades.values()))
        print(f"   {n_comunidades} comunidades detectadas")
    except:
        print("   No se pudo detectar comunidades")
    
    print("   " + "-" * 70)
    
    # Visualizaciones
    red.visualizar_red(mostrar_comunidades=False)
    red.visualizar_distribucion_grados()
    
    # Guardar red
    red.guardar_red('red_hibrida.gexf')
    
