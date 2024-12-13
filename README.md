# nn-rpi-logic-gates

![GitHub License](https://img.shields.io/github/license/antonioberna/nn-rpi-logic-gates)
![GitHub repo size](https://img.shields.io/github/repo-size/antonioberna/nn-rpi-logic-gates)
![GitHub Created At](https://img.shields.io/github/created-at/antonioberna/nn-rpi-logic-gates)

## Introduction

This morning I woke up with the desire to create a project a little different from the usual. To understand, I'm not talking about the classic neural network that returns random results (on the terminal) because that would just be boring.
I felt like combining the power of Machine Learning with Electronics and this is the result: creating a project that is based on training a simple neural network to recognize logic gates in Boole's algebra. But in this way the project would have been based only on a simple (depending on your point of view) math problem. Then I asked myself: "What if we used Electronics to visualize the output of the neural network training?". Specifically using 4 led diodes (each representing one of the outputs of a single logic port) and running the code on my Raspberry Pi 3 could come up with a really interesting project!

## Logic Gates

First I give a brief review for those who have no idea what just the logic gates of boolean algebra.

Boole's algebra logic gates are fundamental elements in digital electronics used to manipulate binary signals, represented as truth values ($0$ or $1$). Logic gates take one or more inputs and produce an output according to the rules of Boole's algebra.

There are several types of logic gates, of which the most common are:

<p align="center">
    <img src=".github/imgs/logic-gates.png">
</p>

These logic gates can be combined to create more complex circuits and perform more sophisticated logic operations. For example, the AND, OR, and NOT gates can be combined to create an XOR (exclusive OR) logic gate, a NAND (NOT-AND) logic gate, a NOR (NOT-OR) logic gate, and many others.

## Mini docs

In this section we see step by step how to replicate the project on your own Raspberry Pi!

First of all you need to make sure you have Raspbian operating system installed on your Raspberry Pi, click on the link to download [Raspberry Pi Imager](https://www.raspberrypi.com/software/) (if you have any problems, don't hesitate to contact me). I won't go into the details of setting up the operating system on a Raspberry Pi in this tutorial, so I'll assume you just plug the Raspberry Pi into a power outlet to boot Raspbian OS.

Generally to use a Raspberry Pi we need a mouse and keyboard, but in this case I leave you the following link which explains how to [enable the connection with the SSH protocol](https://roboticsbackend.com/enable-ssh-on-raspberry-pi-raspbian/) so that we can connect to the Raspberry Pi terminal even if we are on a other computer (clearly you need to connect the Raspberry Pi to the same wifi network as the computer on which you decide to use the SSH protocol).

At this point we need to look up the IP address of the Raspberry Pi when it's connected on the same wifi network as ours and [there are various ways to do it](https://www.raspberrypi.com/documentation/computers/remote-access.html).

I personally use the following command:

```
nmap -sn <ip>/24
```

where `<ip>` can be determined in the following ways:
- on Linux using the command `ip route | grep default`
- on macOS using the command `netstat -nr | grep default`

>  [!NOTE]
> `nmap` is a very famous tool for carrying out port scanning, i.e. aimed at identifying open ports on a target computer or even on ranges of IP addresses, in order to determine which network services are available. For more information [click here](https://nmap.org/book/man.html).

In short, depending on the type of method you decide to use, the Raspberry Pi always has the same IP address while remaining connected to the wifi network. So if for example I use `nmap -sn <ip>/24` among the list of devices connected to the wifi network we get:

```
...
Nmap scan report for raspberrypi (192.168.1.18)
...
```

where `raspberrypi` represents the `hostname` of the Raspberry Pi. Now that we have obtained the IP address we can connect to the Raspberry Pi via the SSH protocol with the following command:

```
ssh pi@192.168.1.18
```

then, by entering the Raspberry Pi login `password`, hopefully we will end up in the following terminal:

```
pi@raspberrypi:~ $
```
At this point we have to copy the project folder (`nn-rpi-logic-gates/`) located on our computer into the Raspberry Pi using the handy `scp` command as follows:

```
scp -r ~/Desktop/nn-rpi-logic-gates pi@192.168.1.18:/home/pi
```

> [!NOTE]
> Clearly to run the `scp` command you must have first cloned this repository on your computer with the command `git clone`.

Well, now that the code is ready to run we just have to take care of the assembly diagram as follows:

<p align="center">
    <img src=".github/imgs/project.png" width="500">
</p>

Unfortunately, using the Raspberry Pi it is more difficult to know the pinout of the board (unlike Arduino). For this reason, if we go back to the terminal with the SSH connection active and use the `pinout` command we get:

```
,--------------------------------.
| oooooooooooooooooooo J8     +====
| 1ooooooooooooooooooo        | USB
|                             +====
|      Pi Model 3B  V1.2         |
|      +----+                 +====
| |D|  |SoC |                 | USB
| |S|  |    |                 +====
| |I|  +----+                    |
|                   |C|     +======
|                   |S|     |   Net
| pwr        |HDMI| |I||A|  +======
`-| |--------|    |----|V|-------'

Revision           : a22082
SoC                : BCM2837
RAM                : 1GB
Storage            : MicroSD
USB ports          : 4 (of which 0 USB3)
Ethernet ports     : 1 (100Mbps max. speed)
Wi-fi              : True
Bluetooth          : True
Camera ports (CSI) : 1
Display ports (DSI): 1

J8:
   3V3  (1) (2)  5V    
 GPIO2  (3) (4)  5V    
 GPIO3  (5) (6)  GND   
 GPIO4  (7) (8)  GPIO14
   GND  (9) (10) GPIO15
GPIO17 (11) (12) GPIO18
GPIO27 (13) (14) GND   
GPIO22 (15) (16) GPIO23
   3V3 (17) (18) GPIO24
GPIO10 (19) (20) GND   
 GPIO9 (21) (22) GPIO25
GPIO11 (23) (24) GPIO8 
   GND (25) (26) GPIO7 
 GPIO0 (27) (28) GPIO1 
 GPIO5 (29) (30) GND   
 GPIO6 (31) (32) GPIO12
GPIO13 (33) (34) GND   
GPIO19 (35) (36) GPIO16
GPIO26 (37) (38) GPIO20
   GND (39) (40) GPIO21

For further information, please refer to https://pinout.xyz/
```

> [!NOTE]
> In particular, if you look at the code present in the `src/main.py` file, you will find the `led_pins` list in which the pins on which I connected the 4 led diodes are saved. Basically the pins to refer to, from the point of view of the code, are those that are shown between the round brackets.

Perfect, now we just have to run the code to see how the neural network is trained based on the logic gate that is selected.

> [!WARNING]
> Since the neural network uses the `numpy` library, we can install it using the command `sudo apt install python3-numpy`. It is preferable to use the `apt` package manager instead of `pip` to avoid problems due to the possible breakage of the Python package manager. Also if you do not have `pip` on your Raspberry Pi, you can install it using the command `sudo apt install python3-pip`. In case of problems use the command `sudo apt-get install --fix-missing python3-pip`.

> [!WARNING]
> The `numpy` library is not installed in a separate virtual environment because otherwise there would be compatibility problems with the `RPi.GPIO` library which is fundamental for managing LED diodes.

In particular, from the terminal with the active SSH session we use the following command:

```
python src/main.py
```

In order to obtain the following outputs:

```
AND model in progress.
[0 0] -> 0.0016023969766617023
[0 1] -> 0.03462897741415107
[1 0] -> 0.041200871702425125
[1 1] -> 0.9426577317372801

OR model in progress.
[0 0] -> 0.06636795579393338
[0 1] -> 0.9662234138295764
[1 0] -> 0.9641925876178026
[1 1] -> 0.97962216640267

NAND model in progress.
[0 0] -> 0.9974056365462818
[0 1] -> 0.9574790586890473
[1 0] -> 0.9666920991443287
[1 1] -> 0.053627724389366874

NOR model in progress.
[0 0] -> 0.9505501441258005
[0 1] -> 0.028460886278734884
[1 0] -> 0.02732870686749378
[1 1] -> 0.008136723247384976

XOR model in progress.
[0 0] -> 0.12040913688930115
[0 1] -> 0.9037882317417703
[1 0] -> 0.9114282554472802
[1 1] -> 0.07309295453195162
```

Since it is difficult for me to show the sequence of led lighting for each logic gate, we modify the `.github/training/data.json` file leaving only the XOR logic gate (which is the most difficult to train):

```
{	
    "training_XOR": [
        {"input": [0, 0], "output": 0},
        {"input": [0, 1], "output": 1},
        {"input": [1, 0], "output": 1},
        {"input": [1, 1], "output": 0}
    ]
}
```

In this way, using the `python src/main.py` command again we get:

<p align="center">
    <img src=".github/imgs/xor.png" width="300">
</p>

## Mathematics behind Machine Learning model

For more information on the mathematics behind the neural network model, I refer you to the [pdf](.github/pdf/deck.pdf) that I'm preparing.

