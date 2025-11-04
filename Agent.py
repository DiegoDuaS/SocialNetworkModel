import numpy as np
from enum import Enum
from typing import Optional, Dict

class EstadoAgente(Enum):
    """Estados posibles del modelo SDN"""
    SUSCEPTIBLE = 'S'
    DESINFORMADO = 'D'
    NO_INVOLUCRADO = 'N'

class TipoAgente(Enum):
    """Tipos de agentes según comportamiento"""
    CREDULO = 'credulo'          # Alta susceptibilidad
    ESCEPTICO = 'esceptico'      # Baja susceptibilidad
    VERIFICADOR = 'verificador'  # Muy baja susceptibilidad, alta corrección

class Agent:
    """
    Clase que representa un agente/usuario en la red social.
    Implementa el modelo SDN (Susceptible-Desinformado-No involucrado).
    """
    
    def __init__(self, id_agente: int, tipo: TipoAgente = None, seed: int = None):
        """
        Inicializa un agente con sus características heterogéneas.
        
        Args:
            id_agente: Identificador único del agente
            tipo: Tipo de agente (CREDULO, ESCEPTICO, VERIFICADOR)
            seed: Semilla para reproducibilidad (opcional)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Identificación
        self.id = id_agente
        self.estado = EstadoAgente.SUSCEPTIBLE
        
        # CARACTERÍSTICAS HETEROGÉNEAS (del PDF)
        self.susceptibilidad = 0.0  # s_i ∈ [0,1]
        self.actividad = 0.0        # a_i ∈ [0,∞]
        self.interes = 0.0          # r_i ∈ [0,1]
        
        # Tipo de agente
        self.tipo = tipo if tipo else self._asignar_tipo_aleatorio()
        
        # Atributos demográficos (opcionales pero recomendados)
        self.edad = None
        self.genero = None
        self.orientacion_politica = None  # Izquierda/Centro/Derecha
        
        # Flags especiales
        self.es_super_propagador = False
        self.es_hub = False
        
        # Inicializar características según tipo
        self._inicializar_caracteristicas()
    
    def _asignar_tipo_aleatorio(self) -> TipoAgente:
        """
        Asigna un tipo de agente aleatoriamente.  40% crédulos, 40% escépticos, 20% verificadores
        """
        tipos = [TipoAgente.CREDULO, TipoAgente.ESCEPTICO, TipoAgente.VERIFICADOR]
        probabilidades = [0.4, 0.4, 0.2]
        return np.random.choice(tipos, p=probabilidades)
    
    def _inicializar_caracteristicas(self):
        """
        Inicializa susceptibilidad, actividad e interés según: Susceptibilidad: Beta(α,β) Actividad: Log-Normal(μ,σ) Interés: Beta(α,β)
        """
        if self.tipo == TipoAgente.CREDULO:
            # Alta susceptibilidad
            self.susceptibilidad = np.random.beta(5, 2)  # Sesgado alto
            self.actividad = np.random.lognormal(0, 0.8)
            self.interes = np.random.beta(4, 2)
            
        elif self.tipo == TipoAgente.ESCEPTICO:
            # Baja susceptibilidad
            self.susceptibilidad = np.random.beta(2, 5)  # Sesgado bajo
            self.actividad = np.random.lognormal(0, 0.5)
            self.interes = np.random.beta(3, 3)
            
        elif self.tipo == TipoAgente.VERIFICADOR:
            # Muy baja susceptibilidad, verifica info
            self.susceptibilidad = np.random.beta(1, 8)  # Muy bajo
            self.actividad = np.random.lognormal(0.5, 0.6)  # Más activo
            self.interes = np.random.beta(5, 2)
    
    def marcar_como_super_propagador(self):
        """
        Los súper propagadores tienen:  Mayor actividad (factor 2-5x) Mayor influencia en la red
        """
        self.es_super_propagador = True
        # Multiplicar actividad
        self.actividad *= np.random.uniform(2, 5)
    
    def marcar_como_hub(self):
        """Marca al agente como hub de la red"""
        self.es_hub = True
    
    # Calculo de Probabilidades de transición ********************
     
    # Metodos de transición de estado ********************
   
    # Metodos para obtener información del agente ********************
    
    # Printing Methods ********************
