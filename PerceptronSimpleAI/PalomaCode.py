
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Definición de patrones de entrada y salida

P = np.array([
    [ 1,  1,  1],
    [ 1,  1, -1],
    [ 1, -1,  1],
    [ 1, -1, -1],
    [-1,  1,  1],
    [-1,  1, -1],
    [-1, -1,  1],
    [-1, -1, -1]
], dtype=float)

T = np.array([ 1, 1, 1, 1, 1, 1, -1, 1 ], dtype=float)

num_patrones = P.shape[0]
num_features = P.shape[1]

# 2. Inicialización entre [-1, 1]
np.random.seed(42)           # Reproducibilidad
W = np.random.uniform(-1, 1, size=num_features)
b = np.random.uniform(-1, 1)
alpha = 1.0
max_epocas = 15
error_epocas = []

# Guardamos los valores iniciales (para mostrar en el cuadro)
W_inicial = W.copy()
b_inicial = b

# 3. Confirguracion visual , separamos la interfaz en 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,6))
fig.suptitle('Entrenamiento del Perceptrón Simple - Problema de la Paloma', fontsize=15, fontweight='bold', color='#1E3A8A')

# Panel izquierdo, Evolución del error 
ax1.set_xlim(0, max_epocas)
ax1.set_ylim(0, num_patrones+2)
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Error total', fontsize=12)
ax1.set_title('Evolución del error', fontsize=13, color='#0F172A')
ax1.grid(True)
linea_error, = ax1.plot([], [], 'o-', color='#2563EB', linewidth=2, markersize=6)
texto_info = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=11,
                      verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Panel derecho Diagrama del perceptrón
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Diagrama del Perceptrón', fontsize=13, color='#0F172A')

input_x = 0.1
input_y = [0.75, 0.5, 0.25]
neuron_x = 0.5
neuron_y = 0.5
output_x = 0.9
output_y = 0.5

# Dibujar nodos de entrada
for i, y in enumerate(input_y):
    ax2.add_patch(plt.Circle((input_x, y), 0.04, color='#93C5FD', ec='black', lw=1.5))
    ax2.text(input_x-0.08, y, f'x{i+1}', fontsize=11, weight='bold', va='center')

# Neurona central
neuron = plt.Circle((neuron_x, neuron_y), 0.07, color='#FDE68A', ec='red', lw=2)
ax2.add_patch(neuron)
ax2.text(neuron_x, neuron_y, 'Σ', fontsize=14, ha='center', va='center', weight='bold')

# Salida
ax2.add_patch(plt.Circle((output_x, output_y), 0.04, color='#86EFAC', ec='black', lw=1.5))
ax2.text(output_x+0.05, output_y, 'y', fontsize=12, weight='bold', va='center')

# Líneas y etiquetas de pesos
textos_pesos = []
for i, y in enumerate(input_y):
    ax2.plot([input_x+0.04, neuron_x-0.07], [y, neuron_y], 'k-', lw=1.5)
    textos_pesos.append(ax2.text((input_x+neuron_x)/2 - 0.015, (y+neuron_y)/2,
                                 '', fontsize=10, ha='center',
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')))

# Bias
texto_bias = ax2.text(neuron_x, neuron_y-0.15, '', ha='center', fontsize=11,
                      bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.3'))

# Cuadro de valores iniciales 
texto_iniciales = ax2.text(
    0.5, 0.05,
    f"Valores iniciales:\n"
    f"W1 = {W_inicial[0]:.2f}\n"
    f"W2 = {W_inicial[1]:.2f}\n"
    f"W3 = {W_inicial[2]:.2f}\n"
    f"b  = {b_inicial:.2f}",
    ha='center', va='bottom', fontsize=10,
    bbox=dict(facecolor='#E5E7EB', edgecolor='black', boxstyle='round,pad=0.6')
)

# 4. Entrenamiento por época

epoca_actual = 0
def entrenar_una_epoca():
    global W, b
    error_total = 0
    for p in range(num_patrones):
        x = P[p]
        t = T[p]
        a = np.dot(W, x) + b
        y = 1 if a >= 0 else -1
        e = t - y
        error_total += abs(e)
        if e != 0:
            W += alpha * e * x
            b += alpha * e
    return error_total


# 5. Inicialización de animación
def init():
    linea_error.set_data([], [])
    texto_info.set_text('')
    for txt in textos_pesos:
        txt.set_text('')
    texto_bias.set_text('')
    return linea_error, texto_info, *textos_pesos, texto_bias


# 6. Actualización por época

def update(frame):
    global epoca_actual

    if epoca_actual < max_epocas:
        error_total = entrenar_una_epoca()
        error_epocas.append(error_total)

        # Actualizar gráfica de error
        x_data = np.arange(1, len(error_epocas)+1)
        linea_error.set_data(x_data, error_epocas)
        texto_info.set_text(f"Época: {epoca_actual+1}\nError total: {error_total}")

        # Actualizar diagrama de pesos
        for i, txt in enumerate(textos_pesos):
            txt.set_text(f"w{i+1} = {W[i]:.2f}")
        texto_bias.set_text(f"b = {b:.2f}")

        # Esperar 3 segundos entre épocas
        plt.pause(3)

        if error_total == 0:
            texto_info.set_text(f"Época: {epoca_actual+1}\nError=0 Convergencia completa")
            anim.event_source.stop()

        epoca_actual += 1

    return linea_error, texto_info, *textos_pesos, texto_bias


# 7. Animación
anim = FuncAnimation(
    fig, update, init_func=init, frames=max_epocas,
    interval=500, blit=False, repeat=False
)

plt.tight_layout()
plt.show()


# 8. Resultados finales

print("\n============================================================")
print("PESOS Y BIAS FINALES")
print("============================================================")
print(f"W = {W}")
print(f"b = {b:.2f}\n")
