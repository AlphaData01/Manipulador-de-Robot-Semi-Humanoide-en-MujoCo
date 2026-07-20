from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

CARPETA_PROYECTO = Path(__file__).resolve().parent
RUTA_XML = CARPETA_PROYECTO / "Manipulador_Semi_Humanoide.xml"

# Pausa entre posturas.
PAUSA_ENTRE_MOVIMIENTOS = 0.4

# Repetir indefinidamente.
REPETIR_RUTINA = True


# ============================================================
# POSICIONES DE LA MANO
# ============================================================
#
# IMPORTANTE:
# Los dedos se controlan mediante longitudes de tendones,
# no mediante grados.
#
# En esta mano, normalmente:
#   longitud mayor  -> dedo más abierto
#   longitud menor  -> dedo más cerrado
#
# Si al probar observas que funciona al contrario, simplemente
# intercambia los valores ABIERTO y CERRADO.
# ============================================================

INDICE_ABIERTO = 0.110387
INDICE_CERRADO = 0.058520

MEDIO_ABIERTO = 0.110387
MEDIO_CERRADO = 0.058520

ANULAR_ABIERTO = 0.110387
ANULAR_CERRADO = 0.058520

MENIQUE_ABIERTO = 0.110387
MENIQUE_CERRADO = 0.058520

# Pulgar: separación lateral.
PULGAR_ABDUCCION_ABIERTO = 0.0
PULGAR_ABDUCCION_CERRADO = 70.0

# Tendones del pulgar.
PULGAR_TENDON_1_ABIERTO = 0.038389
PULGAR_TENDON_1_CERRADO = 0.026152

PULGAR_TENDON_2_ABIERTO = 0.112138
PULGAR_TENDON_2_CERRADO = 0.081568


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def obtener_id_actuador(
    modelo: mujoco.MjModel,
    nombre: str,
) -> int:
    """Obtiene el ID de un actuador a partir de su nombre."""

    actuator_id = mujoco.mj_name2id(
        modelo,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        nombre,
    )

    if actuator_id == -1:
        raise ValueError(
            f"No se encontró el actuador '{nombre}' en el XML."
        )

    return actuator_id


def limitar_control(
    modelo: mujoco.MjModel,
    actuator_id: int,
    valor: float,
) -> float:
    """
    Limita un objetivo al ctrlrange definido en el XML.
    Evita mandar posiciones fuera de rango.
    """

    minimo = float(modelo.actuator_ctrlrange[actuator_id, 0])
    maximo = float(modelo.actuator_ctrlrange[actuator_id, 1])

    return float(np.clip(valor, minimo, maximo))


def avanzar_simulacion(
    modelo: mujoco.MjModel,
    datos: mujoco.MjData,
    visor,
    duracion: float,
) -> None:
    """Mantiene la simulación avanzando durante cierto tiempo."""

    tiempo_final = datos.time + duracion

    while visor.is_running() and datos.time < tiempo_final:
        inicio = time.perf_counter()

        mujoco.mj_step(modelo, datos)
        visor.sync()

        restante = modelo.opt.timestep - (
            time.perf_counter() - inicio
        )

        if restante > 0:
            time.sleep(restante)


def mover_suavemente(
    modelo: mujoco.MjModel,
    datos: mujoco.MjData,
    visor,
    objetivos: dict[int, float],
    duracion: float = 2.0,
) -> bool:
    """
    Interpola varios actuadores simultáneamente.

    Devuelve False si el visor se cerró.
    """

    if duracion <= 0:
        raise ValueError("La duración debe ser mayor que cero.")

    objetivos_limitados = {
        actuator_id: limitar_control(
            modelo,
            actuator_id,
            objetivo,
        )
        for actuator_id, objetivo in objetivos.items()
    }

    valores_iniciales = {
        actuator_id: float(datos.ctrl[actuator_id])
        for actuator_id in objetivos_limitados
    }

    pasos = max(
        1,
        int(duracion / modelo.opt.timestep),
    )

    for paso in range(pasos):
        if not visor.is_running():
            return False

        inicio_paso = time.perf_counter()

        proporcion = (paso + 1) / pasos

        # Interpolación smoothstep:
        # arranca y termina suavemente.
        proporcion_suave = (
            3.0 * proporcion**2
            - 2.0 * proporcion**3
        )

        for actuator_id, objetivo in objetivos_limitados.items():
            inicial = valores_iniciales[actuator_id]

            datos.ctrl[actuator_id] = (
                inicial
                + (objetivo - inicial) * proporcion_suave
            )

        mujoco.mj_step(modelo, datos)
        visor.sync()

        restante = modelo.opt.timestep - (
            time.perf_counter() - inicio_paso
        )

        if restante > 0:
            time.sleep(restante)

    return True


# ============================================================
# POSTURAS DE LA MANO
# ============================================================

def objetivos_mano_abierta(motores: dict[str, int]) -> dict[int, float]:
    """Devuelve los controles correspondientes a una mano abierta."""

    return {
        motores["indice"]: INDICE_ABIERTO,
        motores["medio"]: MEDIO_ABIERTO,
        motores["anular"]: ANULAR_ABIERTO,
        motores["menique"]: MENIQUE_ABIERTO,
        motores["pulgar_abduccion"]: PULGAR_ABDUCCION_ABIERTO,
        motores["pulgar_tendon_1"]: PULGAR_TENDON_1_ABIERTO,
        motores["pulgar_tendon_2"]: PULGAR_TENDON_2_ABIERTO,
    }


def objetivos_mano_cerrada(motores: dict[str, int]) -> dict[int, float]:
    """Devuelve los controles correspondientes a una mano cerrada."""

    return {
        motores["indice"]: INDICE_CERRADO,
        motores["medio"]: MEDIO_CERRADO,
        motores["anular"]: ANULAR_CERRADO,
        motores["menique"]: MENIQUE_CERRADO,
        motores["pulgar_abduccion"]: PULGAR_ABDUCCION_CERRADO,
        motores["pulgar_tendon_1"]: PULGAR_TENDON_1_CERRADO,
        motores["pulgar_tendon_2"]: PULGAR_TENDON_2_CERRADO,
    }


def objetivos_pinza(motores: dict[str, int]) -> dict[int, float]:
    """
    Gesto de pinza:
    índice y pulgar cierran; los otros dedos permanecen abiertos.
    """

    return {
        motores["indice"]: INDICE_CERRADO,
        motores["medio"]: MEDIO_ABIERTO,
        motores["anular"]: ANULAR_ABIERTO,
        motores["menique"]: MENIQUE_ABIERTO,
        motores["pulgar_abduccion"]: PULGAR_ABDUCCION_CERRADO,
        motores["pulgar_tendon_1"]: PULGAR_TENDON_1_CERRADO,
        motores["pulgar_tendon_2"]: PULGAR_TENDON_2_CERRADO,
    }


def objetivos_saludo(motores: dict[str, int]) -> dict[int, float]:
    """
    Gesto sencillo:
    índice y medio abiertos, anular y meñique cerrados.
    """

    return {
        motores["indice"]: INDICE_ABIERTO,
        motores["medio"]: MEDIO_ABIERTO,
        motores["anular"]: ANULAR_CERRADO,
        motores["menique"]: MENIQUE_CERRADO,
        motores["pulgar_abduccion"]: PULGAR_ABDUCCION_CERRADO,
        motores["pulgar_tendon_1"]: PULGAR_TENDON_1_CERRADO,
        motores["pulgar_tendon_2"]: PULGAR_TENDON_2_CERRADO,
    }


def objetivos_puno_parcial(motores: dict[str, int]) -> dict[int, float]:
    """Cierra la mano parcialmente."""

    return {
        motores["indice"]: (
            INDICE_ABIERTO + INDICE_CERRADO
        ) / 2.0,

        motores["medio"]: (
            MEDIO_ABIERTO + MEDIO_CERRADO
        ) / 2.0,

        motores["anular"]: (
            ANULAR_ABIERTO + ANULAR_CERRADO
        ) / 2.0,

        motores["menique"]: (
            MENIQUE_ABIERTO + MENIQUE_CERRADO
        ) / 2.0,

        motores["pulgar_abduccion"]: 35.0,

        motores["pulgar_tendon_1"]: (
            PULGAR_TENDON_1_ABIERTO
            + PULGAR_TENDON_1_CERRADO
        ) / 2.0,

        motores["pulgar_tendon_2"]: (
            PULGAR_TENDON_2_ABIERTO
            + PULGAR_TENDON_2_CERRADO
        ) / 2.0,
    }


# ============================================================
# RUTINA COMPLETA
# ============================================================

def ejecutar_rutina(
    modelo: mujoco.MjModel,
    datos: mujoco.MjData,
    visor,
    motores: dict[str, int],
    numero_ciclo: int,
) -> bool:
    """Ejecuta un ciclo completo del movimiento."""

    print(f"\n========== CICLO {numero_ciclo} ==========")

    # --------------------------------------------------------
    # 1. Postura inicial: brazo abajo y mano abierta
    # --------------------------------------------------------

    print("1. Postura inicial y mano abierta")

    objetivos = {
        motores["hombro_1"]: 0.0,
        motores["hombro_2"]: 0.0,
        motores["bicep_z"]: 0.0,
        motores["brazo"]: 0.0,
    }

    objetivos.update(objetivos_mano_abierta(motores))

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        objetivos,
        duracion=2.5,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        PAUSA_ENTRE_MOVIMIENTOS,
    )

    # --------------------------------------------------------
    # 2. Levantar el brazo
    # --------------------------------------------------------

    print("2. Levantando el brazo")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        {
            motores["hombro_1"]: 45.0,
            motores["hombro_2"]: -45.0,
            motores["brazo"]: 35.0,
        },
        duracion=2.5,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        PAUSA_ENTRE_MOVIMIENTOS,
    )

    # --------------------------------------------------------
    # 3. Cerrar parcialmente la mano
    # --------------------------------------------------------

    print("3. Cerrando parcialmente la mano")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        objetivos_puno_parcial(motores),
        duracion=1.5,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        PAUSA_ENTRE_MOVIMIENTOS,
    )

    # --------------------------------------------------------
    # 4. Cerrar completamente
    # --------------------------------------------------------

    print("4. Cerrando la mano")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        objetivos_mano_cerrada(motores),
        duracion=1.7,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        0.8,
    )

    # --------------------------------------------------------
    # 5. Rotar el brazo con el puño cerrado
    # --------------------------------------------------------

    print("5. Rotando el brazo con el puño cerrado")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        {
            motores["bicep_z"]: 55.0,
            motores["hombro_1"]: 60.0,
            motores["hombro_2"]: -70.0,
            motores["brazo"]: 65.0,
        },
        duracion=2.5,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        PAUSA_ENTRE_MOVIMIENTOS,
    )

    # --------------------------------------------------------
    # 6. Abrir la mano
    # --------------------------------------------------------

    print("6. Abriendo la mano")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        objetivos_mano_abierta(motores),
        duracion=1.7,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        0.6,
    )

    # --------------------------------------------------------
    # 7. Hacer gesto de pinza
    # --------------------------------------------------------

    print("7. Haciendo gesto de pinza")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        objetivos_pinza(motores),
        duracion=1.5,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        1.0,
    )

    # --------------------------------------------------------
    # 8. Mover brazo durante la pinza
    # --------------------------------------------------------

    print("8. Movimiento coordinado con pinza")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        {
            motores["hombro_1"]: 30.0,
            motores["hombro_2"]: -110.0,
            motores["bicep_z"]: -45.0,
            motores["brazo"]: 80.0,
        },
        duracion=3.0,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        PAUSA_ENTRE_MOVIMIENTOS,
    )

    # --------------------------------------------------------
    # 9. Gesto con dos dedos
    # --------------------------------------------------------

    print("9. Gesto con índice y medio abiertos")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        objetivos_saludo(motores),
        duracion=1.5,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        1.0,
    )

    # --------------------------------------------------------
    # 10. Abrir la mano y extender el brazo
    # --------------------------------------------------------

    print("10. Extendiendo el brazo y abriendo la mano")

    objetivos = {
        motores["hombro_1"]: 20.0,
        motores["hombro_2"]: -35.0,
        motores["bicep_z"]: 0.0,
        motores["brazo"]: 10.0,
    }

    objetivos.update(objetivos_mano_abierta(motores))

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        objetivos,
        duracion=3.0,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        0.8,
    )

    # --------------------------------------------------------
    # 11. Movimiento lateral del brazo
    # --------------------------------------------------------

    print("11. Movimiento lateral")

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        {
            motores["hombro_1"]: 70.0,
            motores["hombro_2"]: -90.0,
            motores["bicep_z"]: 70.0,
            motores["brazo"]: 45.0,
        },
        duracion=3.0,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        PAUSA_ENTRE_MOVIMIENTOS,
    )

    # --------------------------------------------------------
    # 12. Cerrar y abrir dos veces
    # --------------------------------------------------------

    for repeticion in range(2):
        print(
            f"12.{repeticion + 1}. "
            "Cerrando y abriendo la mano"
        )

        if not mover_suavemente(
            modelo,
            datos,
            visor,
            objetivos_mano_cerrada(motores),
            duracion=1.0,
        ):
            return False

        if not mover_suavemente(
            modelo,
            datos,
            visor,
            objetivos_mano_abierta(motores),
            duracion=1.0,
        ):
            return False

    # --------------------------------------------------------
    # 13. Regresar a la posición inicial
    # --------------------------------------------------------

    print("13. Regresando a la postura inicial")

    objetivos = {
        motores["hombro_1"]: 0.0,
        motores["hombro_2"]: 0.0,
        motores["bicep_z"]: 0.0,
        motores["brazo"]: 0.0,
    }

    objetivos.update(objetivos_mano_abierta(motores))

    if not mover_suavemente(
        modelo,
        datos,
        visor,
        objetivos,
        duracion=3.5,
    ):
        return False

    avanzar_simulacion(
        modelo,
        datos,
        visor,
        1.0,
    )

    print(f"Ciclo {numero_ciclo} terminado.")

    return True


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    if not RUTA_XML.exists():
        raise FileNotFoundError(
            f"No se encontró el XML:\n{RUTA_XML}"
        )

    print(f"Cargando modelo:\n{RUTA_XML}")

    modelo = mujoco.MjModel.from_xml_path(
        str(RUTA_XML)
    )

    datos = mujoco.MjData(modelo)

    # --------------------------------------------------------
    # Obtener IDs de los actuadores
    # --------------------------------------------------------

    motores = {
        # Brazo
        "hombro_1": obtener_id_actuador(
            modelo,
            "motor_hombro_1",
        ),
        "hombro_2": obtener_id_actuador(
            modelo,
            "motor_hombro_2",
        ),
        "bicep_z": obtener_id_actuador(
            modelo,
            "motor_bicep_z",
        ),
        "brazo": obtener_id_actuador(
            modelo,
            "motor_brazo",
        ),

        # Dedos
        "indice": obtener_id_actuador(
            modelo,
            "right_index_A_tendon",
        ),
        "medio": obtener_id_actuador(
            modelo,
            "right_middle_A_tendon",
        ),
        "anular": obtener_id_actuador(
            modelo,
            "right_ring_A_tendon",
        ),
        "menique": obtener_id_actuador(
            modelo,
            "right_pinky_A_tendon",
        ),

        # Pulgar
        "pulgar_abduccion": obtener_id_actuador(
            modelo,
            "right_thumb_A_cmc_abd",
        ),
        "pulgar_tendon_1": obtener_id_actuador(
            modelo,
            "right_th1_A_tendon",
        ),
        "pulgar_tendon_2": obtener_id_actuador(
            modelo,
            "right_th2_A_tendon",
        ),
    }

    print("\nActuadores encontrados correctamente:")

    for nombre, actuator_id in motores.items():
        minimo = modelo.actuator_ctrlrange[
            actuator_id,
            0,
        ]

        maximo = modelo.actuator_ctrlrange[
            actuator_id,
            1,
        ]

        print(
            f"  {nombre:20s} "
            f"ID={actuator_id:2d} "
            f"rango=[{minimo:.6f}, {maximo:.6f}]"
        )

    # --------------------------------------------------------
    # Estado inicial
    # --------------------------------------------------------

    controles_iniciales = {
        motores["hombro_1"]: 0.0,
        motores["hombro_2"]: 0.0,
        motores["bicep_z"]: 0.0,
        motores["brazo"]: 0.0,
    }

    controles_iniciales.update(
        objetivos_mano_abierta(motores)
    )

    for actuator_id, objetivo in controles_iniciales.items():
        datos.ctrl[actuator_id] = limitar_control(
            modelo,
            actuator_id,
            objetivo,
        )

    mujoco.mj_forward(modelo, datos)

    # --------------------------------------------------------
    # Iniciar visor y bucle
    # --------------------------------------------------------

    with mujoco.viewer.launch_passive(
        modelo,
        datos,
    ) as visor:

        print("\nSimulación iniciada.")
        print("La rutina se repetirá automáticamente.")
        print("Cierra la ventana de MuJoCo para terminar.")

        avanzar_simulacion(
            modelo,
            datos,
            visor,
            duracion=1.5,
        )

        numero_ciclo = 1

        while visor.is_running():
            completado = ejecutar_rutina(
                modelo,
                datos,
                visor,
                motores,
                numero_ciclo,
            )

            if not completado:
                break

            numero_ciclo += 1

            if not REPETIR_RUTINA:
                print("Rutina terminada sin repetición.")

                while visor.is_running():
                    avanzar_simulacion(
                        modelo,
                        datos,
                        visor,
                        duracion=0.1,
                    )

                break


if __name__ == "__main__":
    main()