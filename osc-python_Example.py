# Author: Milton Candela (https://github.com/milkbacon)
# Date: February 2023

# Script de python que envía una palabra cada cierta cantidad de segundos, a través del protocolo OSC.

import argparse
from time import sleep
from random import randint
from pythonosc import udp_client

# Banco de palabras de dónde se obtendrán de manera aleatoria
word_bank = ['love', 'passion', 'sadness', 'joy', 'emotion', 'sea', 'storm', 'eye', 'body', 'mind', 'tears', 'spirits',
             'desire', 'blood', 'pain', 'reason', 'landscape', 'nature', 'memory']

IP = "10.22.235.48"  # Escribe el IP aquí
PORT = 5000  # Escribe el puerto aquí


def send_words(word_bank, client, t_wait):
    """
    :param list word_bank: Lista de palabras de donde se escogerá un mensaje.
    :param python-osc.udp_client client: Cliente OSC por donde se enviarán los datos.
    :param list t_wait: Rango de tiempos de espera entre un mensaje y otro [t_ini, t_fin].
    :return void: No regresa nada debido a que el flujo se queda loopeado indefinidamente en el ciclo while.
    """

    l_past = []  # Lista de ID enviados previamente

    while True:
        word_id = randint(0, len(word_bank) - 1)  # Selecciona de manera aleatoria un ID para el banco de palabras

        # Limpia la lista de los mencionados previamente, esto en caso de que ya hayan sido mencionados todos
        if len(l_past) == len(word_bank):
            l_past = []

        # Verifica si el ID ya ha sido seleccionado previamente
        if word_id not in l_past:
            l_past.append(word_id)  # Si el ID no se ha mencionado, entonces se include en la lista "l_past"
            client.send_message("/word", word_bank[word_id])  # Envía el mensaje a través de la dirección "/word"
            sleep(randint(t_wait[0], t_wait[1]))  # Espera una cierta cantidad de segundos entre envios de mensaje


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=IP, help="The ip of the OSC server")

    # Para el parser en el argumento puerto, es importante cambiar "type" dependiendo del tipo de variable que se
    # enviará, en este caso es del tipo "string", por lo que en "type" ponemos "str", pero puede ser "int", "float".
    parser.add_argument("--port", type=str, default=PORT, help="The port the OSC server is listening on")

    args = parser.parse_args()
    client = udp_client.SimpleUDPClient(args.ip, args.port)  # Genera el cliente con base en los argumentos

    send_words(word_bank=word_bank, client=client, t_wait=[5, 30])  # Llama la función para proporcionar mensajes
