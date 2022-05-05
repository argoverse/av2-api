# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Clustering of TbV vehicle logs, by spatial location (scene).

Note:
   - Each log within a cluster shares some significant visual overlap with other logs within its cluster.
   - These are not before/after pairs. In some cases, all logs in a cluster may be "after" a change.
   - Each cluster has at least one log in the val or test set.
   - Logs within each cluster are provided in chronological order.
"""
from typing import Dict, Final, Tuple

ATX_CLUSTERS: Final[Dict[int, Tuple[str, ...]]] = {
    1: (
        "yrQZTJSIjWgFJ3CNE9H4KL1p1QgFmrfb__Spring_2020",
        "Fu5LS0Rw8Fu80C1NagC4A3deS8mgVt7b__Summer_2020",
    ),
    2: (
        "jQb5qhBrBcqhmrX25s7otzQWolLyYfAY__Summer_2020",
        "2p9Pwfm1KbMltk4vpv6ZWlJmZAXi5pvh__Summer_2020",
    ),
    3: ("X4zQYzXVLTQ6zlvX85RNExX833eTCXBy__Summer_2020",),
    4: ("fnX5fTajNRYrkvTe3kgPEcqGND6xoJ83__Summer_2020",),
    5: (
        "WfnwYRUwXgzuvvYBNIV5ffzVGN7ARkRn__Autumn_2020",
        "8326651699a845b8ad0084e6871745f4__Autumn_2020",
    ),
    6: ("d6GIXhzvM5asQxBYlp9K2oXC0yS9CTc3__Winter_2021",),
    7: ("yoVO4riYwUN33RkD1jzX4UNaLaPJ4WwY__Winter_2021",),
    8: ("QCgniBLhqw9yE7xHcNJ6Eud6cvxx9Q2c__Autumn_2020",),
    9: ("ICcruTEH4V7xOTkYXre6rY6CI1rOIwym__Winter_2021",),
    10: ("HzDHGixXphRTwTPVaiBa7r5i0pmw4uO2__Summer_2020",),
    11: ("af170aac84653d7b82c564147e94af7d__Summer_2020",),
}

PIT_CLUSTERS: Final[Dict[int, Tuple[str, ...]]] = {
    1: (
        "0WHfUzbQi8mopDFgQ7BH7lGjgCEygaoN__Summer_2020",
        "yMCh1ZqomU1kChqq9HHTKdrtrFzUH9Pq__Summer_2020",
    ),
    2: (
        "is199hEjAfihF1BTSf3s0gDwVwBemIdy__Spring_2020",
        "7d749a7b99f243f5adb64125806f92e3__Summer_2020",
        "1SKq8bGqPRZq6PUbFgTwVLyMZEwtiD2S__Summer_2020",
        "ofvDkF5aBpNzn7kIuvlaW8QLhxuJ5X3A__Summer_2020",
        "chLbAIiMJCL96sLQSvAKV5lAt05GU4d3__Summer_2020",
        "fzt2N0vjcQ0qiV5APbtelI6beq8tMtEE__Summer_2020",
        "FAgoA7JrnYrC0WJTE3eQZSf2R5r1b3On__Summer_2020",
        "XiTmGzRNQHk6B3NOeOmwY0ruV72e3nZH__Summer_2020",
        "4jptcpHjg76VmiNmHtb8x2qKkIBeBeBW__Summer_2020",
    ),
    3: (
        "8zeNElNT6mNvKxUAzB0IooE0qo8I75e9__Summer_2020",
        "lLnldXKakQk3B27i01yaeaW8Fq3AOzBz__Summer_2020",
    ),
    4: (
        "5KfD7QylF1vufySVeCYZIBlknh2BV6vk__Spring_2020",
        "35f6bbd914d34c4b9a5c63d86f77aece__Spring_2020",
        "pm7wjqKSfq0SFsAcF40rummG3Sa1bqMe__Summer_2020",
    ),
    5: (
        "9hOChFGYejymWsN5rWVEaQccfsBjMHd5__Summer_2020",
        "Jp6Jnwfg3lmF9h1lRugGBf6EV4cxFUlU__Summer_2020",
        "g2sEJ9UhzZfXm58nA27c6H5kS9IcLm8J__Autumn_2020",
        "L8IgAsQLP2oj45QCSU95d5kgl5teOoZx__Autumn_2020",
    ),
    6: (
        "AmwBapoTe2LDA2QLE1UfndhzXvW5qYx7__Summer_2020",
        "0CjqAXeTID58UXtezwdAag5zt6bpsKFp__Summer_2020",
        "EYYePho1wT9s2e4CH7DdCrJYBRB2Syjn__Summer_2020",
        "Uyc6uDJkss47KUDFwhy2NZp6pFluCJ0j__Winter_2021",
    ),
    7: ("6NPQAzBEDnwOp6d2lSXmqjRITGdwQmIY__Summer_2020",),
    8: ("X3e1Hd8bYrP5JU0uBf0HMTx6zomMHyiC__Summer_2020",),
    9: (
        "pbADFDy5ElABBs4vTFGnGtkQjTqIDKyD__Summer_2020",
        "pjKNfNqBGnvRZ4YoXHdwZAxxVojxSQgc__Autumn_2020",
        "79TuaBnpYGOFKZAT1ZcmkUNEjHJtiiup__Autumn_2020",
    ),
    10: (
        "t2EYzOIfnpUeOGaO5kEQF4c2ogSXmVcQ__Summer_2020",
        "15eb911365cc45e897494de6a5356092__Summer_2020",
    ),
    11: (
        "n9l6ATRvHumaZKpfteZTdVYKLEFCNVcS__Summer_2020",
        "yrSiNEmxurcQKfJZ7preY9upbufo7MQ5__Summer_2020",
    ),
    12: ("HYKgsU3gkodpc5EyJdDRR4XtReHb7zup__Summer_2020",),
    13: (
        "VrKB4vQ6rn05E6lC7C6KNSWXRbqImakD__Summer_2020",
        "mx4SgGLgvzlElDsIKMzAd71SAo47ktQ9__Summer_2020",
        "Ixz1DHSjVyEyCNM7sF0EfXy1HxgDgjtP__Winter_2021",
    ),
    14: (
        "92meA40rbhPx3yQSZv8NjKPj7oTi6ipn__Spring_2020",
        "nKNY9xmAgQdC51KKhyy59VJzPgrCWmS6__Spring_2020",
        "QsOXYRRypDjS4vjX2CW1h6kBgjodHjBU__Spring_2020",
        "G1j077kzVj2YDDCGPfdJN5emwtvS6Ffa__Spring_2020",
        "8qQumE8wEiK3wFWRGfGVzLEKqwebUKh6__Summer_2020",
        "gjBPR5cBwUrfFlGvTokRAEXbsEUQjLsY__Summer_2020",
        "4CRsqbn3dhINweAM1ZoMMdwcrIXMYTYh__Summer_2020",
        "exmmrXSlFZJncbjug3PqKJPTmt0NkW5k__Summer_2020",
        "Jd8kn8Nr0QSFCmXA4nBo0wtk2Ylii3Zp__Summer_2020",
        "8qvpfIhDIsg38bFFmyoW2nmUtiWZcPmO__Summer_2020",
        "DOM7SXgdXj6ZLWlRZdc9Y4PH4QgbToAq__Summer_2020",
        "csZYLQhiOECYXex5uILcZoUgBL23MGHf__Autumn_2020",
    ),
    15: ("GNXSrcIoLYHLwFveJUWWdidRgPl1BXBA__Autumn_2020",),
    16: ("Q1KdcvPbyQ7gAFmXUxYzpSfgdfHQWqkv__Autumn_2020",),
    17: ("WqwX2CqJKhhq9F7AJio4X6BqpS1sl57T__Winter_2021",),
    18: ("AElCQ5nI41SJfDxI6BiNJcxvCmV4rRMK__Winter_2021",),
    19: (
        "7K4i22BwSwR2AHsk0novsZmKk74oJlBd__Summer_2020",
        "7NumovZt9STex6xdUaxPuHFiYeFaIGtw__Autumn_2020",
    ),
    20: ("QyCieQww0D5YwoRGcLpjd3pIr0IsGOhX__Autumn_2020",),
    21: ("pe9dhAyNhN9Qvywcw1Vci366LeTeKhIc__Autumn_2020",),
    22: ("VEZpQwdWWLEHGow9KZp5ivZQ5Wq6jn9m__Autumn_2020",),
    23: ("93fSHwp6yVp92cnHoTCGnr0X5mOKb4Y6__Winter_2021",),
    24: ("7IL65wUk7aHOux3AeAdF1XUKtoEPoUjt__Winter_2021",),
}

MIA_CLUSTERS: Final[Dict[int, Tuple[str, ...]]] = {
    1: ("pha5AK0yAM7zlDeBTzdgbll7vSud231q__Spring_2020",),
    2: ("ujhUH2flle6ctyPZawAe9GiSzoeqkAJX__Summer_2020",),
    3: ("jd4WeIiaSQrxZmoHyIhpGj4yR0j1GKnO__Summer_2020",),
    4: ("vbiL75K7RYZmeONb9dJogRi5fsyqwNt1__Summer_2020",),
    5: ("LEIaZYYGdIH5PxnCG6FcFrc0e5PHGWK5__Autumn_2020",),
    6: (
        "vrh2qhEhbgS1diwdmnsUBwQggBAVLIjg__Summer_2020",
        "pImCkff0HddpkAiIAh9QSKpOiFpkD5wz__Summer_2020",
        "7103cbc277fe4728b357bb95b1d17588__Summer_2020",
    ),
    7: (
        "g49jZtybu3cARdzTRtsP2fo8Gdc7S8oX__Summer_2020",
        "2d32edc25f65413fb45942085dfdad2a__Summer_2020",
    ),
    8: (
        "vqHJUhcAKypL4OAvR2ubV5cY1dHLfMm2__Summer_2020",
        "6EJwfT8vvFouPlGvQQp19O0NEpfekc36__Summer_2020",
    ),
    9: (
        "TjYDxQLByFX0Go3Wj18JkY5gCVePA0Xl__Spring_2020",
        "Hl998Yzj5om0qqGHGkMFI15srslWo2l0__Spring_2020",
        "toxA95laeQskeSQ7zCwlcpt2z8kteGDE__Spring_2020",
        "O4zg0xhmQDj1qas2qgZcn5a7LiYjLgMT__Spring_2020",
        "50OQgpEJys7utcy6xf36RZmRrW6RjzhK__Spring_2020",
        "sHD672b17MrFJXOfx49DNXq84UdDgtkU__Spring_2020",
        "KCAFVr17ic63507rz5HyEBH9YU130yOY__Spring_2020",
        "xrCsw3rTAajrGMPzXfRTZ8Gv4MWoGqmh__Summer_2020",
        "bjalGQhAZWMLh50K0poYHX6GcXxnJPom__Summer_2020",
        "zU1PSul4gjlW868IcKBXy4nNrfowwG7K__Summer_2020",
        "9nS3Yf7Aj93OuzMPt9toWEQ8bF7Cs0PT__Summer_2020",
    ),
    10: (
        "rnMrMxQir82A9OcpbJzsDouA2485Y4aH__Spring_2020",
        "3p7HJK6MSYl6Zm6YHNhMStAcbiu7G3sg__Spring_2020",
    ),
    11: (
        "gs1B8ZCv7DMi8cMt5aN5rSYjQidJXvGP__Summer_2020",
        "zFb6NmwKZQ8zWRP2G0WM75YqUVTrhZ7p__Autumn_2020",
    ),
    12: ("McHe1Ns3x3zlrt3SoYFup2QU4X3fgUyT__Summer_2020",),
    13: (
        "SP1puuZR5pn6JY3eT7qIU9XQbw3qh2iK__Summer_2020",
        "20dec8c7eaa54f6c98b6ab29382464f1__Summer_2020",
        "uAWyh9vy6ABsSiN5LmAMgX4V0koRNEaW__Summer_2020",
        "766c171f7fde4e1a80426edf5c527d13__Summer_2020",
    ),
    14: ("hbpV4oknIcVLpZHSEG1nYu7YXl7AxGEX__Spring_2020",),
    15: ("j1yqGvVV6p3xCOwU0JMm4dhAwxjX1Sfy__Spring_2020",),
    16: ("6946khzM4whpgC8SzZVuy7Sr7DDZSv34__Autumn_2020",),
    17: ("mYgfbdqS00jEhgT9N38MWwZB1Lzjwez7__Winter_2021",),
    18: ("Mp9dxH3TBKOIJ5uhSsUtuYDwicigLqgU__Autumn_2020",),
    19: ("LmcFXsufJFxeWwZMrfHg2XOF0nVtugfs__Summer_2020",),
    20: (
        "7iShen3l9XJokWc7mVxaPTxWaKq8Jjgb__Autumn_2020",
        "LrQRfUKwDViVzVs0wp1VNYDUQcwEgJj3__Autumn_2020",
        "7dcece6971264b7aba6fc90e7f9df8ad__Autumn_2020",
    ),
    21: (
        "xpxgFOU6dHCdsqhWVWTkb5xYTAzxQUwj__Autumn_2020",
        "9qzWKvwnai0YtPDIFkig8cstG4ibyhZa__Autumn_2020",
        "4ab4f1a7ae0146399153d6170bb1dcd7__Autumn_2020",
        "CTZa7zmqmUNsO0ZHYmHoP1EUSu1oT1Qh__Winter_2020",
    ),
    22: ("TQqZLr12AhhPnvRgR5p4czYhOiXqDND8__Autumn_2020",),
    23: ("WzrqGbZA4v2nerkUpM3hbrmfg56eyb61__Autumn_2020",),
    24: (
        "CInXhoNsgVjJ1HhnjfFIXnVPpnfHURRa__Summer_2020",
        "e4f75a5b5d2d4accb458dc829c1e9f3f__Summer_2020",
    ),
    25: (
        "54bc6dbcebfb3fbab5b357f88b4b79ca__Winter_2020",
        "0f0cdd79bc6c35cd9d997ae2fc7e165c__Winter_2020",
        "Aom5hOS7NB2aexV59g91afgp4cLYxZxx__Summer_2020",
    ),
    26: ("2ZDdTQbgvWsyQZGwhg9EkKBRt9nk6fiY__Winter_2021",),
    27: (
        "IFeHPOu37ZylGkr8ntl7vdU7fdRnUkGm__Summer_2020",
        "AVxsoMHjo3GYsKovmgZPku9usXLBqfCP__Autumn_2020",
        "pu5XaiqeMKZBOt1MUZsOXdJPjZqXU5J7__Autumn_2020",
    ),
    28: (
        "LOgutcHPKdh7rA0PLSGWhSqGyiShi3WI__Winter_2021",
        "e55ruBKpsoV2IYOAmgLIEU5u4BcoQTff__Winter_2021",
        "9p0jAvJXOWlcYlWHaHz3MUXVpfdrHN86__Winter_2021",
    ),
    29: ("36guOvKtHGyqGmkQrsTeWCvA8A6084se__Autumn_2020",),
    30: (
        "WKdv3YSBogcfx2ivUoSlNNRwX7gNDxiA__Autumn_2020",
        "D1nzwIZVxEtXaWglThZBaW2jupdU5J89__Winter_2021",
    ),
}

WDC_CLUSTERS: Final[Dict[int, Tuple[str, ...]]] = {
    1: (
        "iq1DbXgcmkBLNjBydqZ2b3uAVCESglqa__Summer_2020",
        "Kyb6Sa3D6O9sbliMdufjWBZp5hOKFAba__Summer_2020",
    ),
    2: (
        "Sek2BIMukaeBarKaxPqPXS4FiKTkrxNB__Summer_2020",
        "B6JWFEqqecVSM0Hxpt9IuZ9Y0YJI0AIm__Summer_2020",
    ),
    3: ("v6E56tiTAAK6x4aPUSb2J0pYf2qFq4r1__Summer_2020",),
    4: ("1mQ4VuN3CZ27Xh7iXUILqKuNw1tciHrv__Summer_2020",),
    5: ("aQC9O61Cahjzjq3IibAWwavhkzpAQ5m4__Summer_2020",),
    6: ("EP9PjbeBPfKRiTfbvKhzzyzjh8KoHDxE__Summer_2020",),
    7: ("0RRXOItl29gFaocqQW297gkzqmpP0mog__Summer_2020",),
    8: ("nrrQP1daZFx5oKBfIcMEt4J004o3QSeK__Summer_2020",),
    9: ("wvWtwTIZ3doB1xqX0j21vr0Qbaxcaoet__Summer_2020",),
    10: (
        "DAkHLZRZorldGbqJfbQ6u5T9dmJgqclO__Spring_2020",
        "LMv8CiEU9Ak8zHO1aKKy6PdQbsTjWUG9__Spring_2020",
        "JhRijIKfPFnA78wiV0mG3Y2tsFEiusj6__Spring_2020",
        "7808aac1740d4ac6b367e6f18e45d0dd__Spring_2020",
    ),
    11: (
        "ZlF2B05IQPcc6xfq4wza7t9XqV1OxYyT__Spring_2020",
        "M9XZ86KEJsQOT64pOQAkvlthhr3Z4Sk3__Spring_2020",
    ),
    12: ("NtMLrGFePjJ1DvFDZG1p9r9DFBf5d4VV__Spring_2020",),
    13: (
        "Kgobyet2FXesqYI3wCkli0ft7Q7t0Plt__Spring_2020",
        "XNiF0SpBmYYMykds22TpRftba3fip6uc__Spring_2020",
    ),
    14: (
        "5egFz1KKT87Q8h42m5cTm2vrzgSYordr__Spring_2020",
        "1818f7906b2643f19dcb778306e63326__Spring_2020",
    ),
    15: ("KkdwO1CyPxSwQ24FbFyCKIbu86HOjYcO__Summer_2020",),
    16: ("MiLpa3m1rDRmAFNDwc1T4kbnhB86Fe5M__Spring_2020",),
    17: ("ZtukedfPrM7jTU0KAhPWlBIHZCUYjRok__Summer_2020",),
    18: ("qTU3OANchNtEPsSuoNbjE3FsbNBvv47T__Autumn_2020",),
    19: ("ESCG9uyqR1lI8Vr5sMyOkUYWEMPrclWn__Autumn_2020",),
    20: ("qT9M5446NgGW5izOozHsSM9gLGyGkD1u__Summer_2020",),
    21: ("vb7eprKk3XG1rwkvfRmZV4ON3cJMFZER__Summer_2020",),
    22: (
        "yAVHpYFQhhWjXlgAWTz8kDsyAQ4iMFMo__Summer_2020",
        "58eb8572cbc7420eaa207e1a3137d6ed__Summer_2020",
    ),
    23: ("1iEFkgp7lcKN3NbORA3RJNXdHOmNVljX__Spring_2020",),
    24: ("IfDKJGa3z4LEFCYj5JSLGEuw9LI4UIlI__Summer_2020",),
    25: ("UaSNn6bBH1RYtTnXKovPAj84xpBeGnDB__Summer_2020",),
    26: ("pemP2CGXlt8bmEFYmpCcI1y3zyY05Jxc__Summer_2020",),
    27: ("UQ6RQlcjsUjzTSHg11glWkaXCabaqkrM__Summer_2020",),
    28: ("RA5rggnWv3hZ6OwKMuRxaiR208i43CV8__Summer_2020",),
    29: (
        "X1yjCkSira47fXPmQVIANIrJ85YWLTUo__Autumn_2020",
        "d200493fbaf44aeebb7e8037a8e22279__Autumn_2020",
    ),
    30: ("pJemcSiXghbfGkHVcEvFppgl5Ye9aMcC__Summer_2020",),
    31: ("r1IDo9jm3BIO5ALqAA7Oz95yK8ZmGATb__Summer_2020",),
    32: ("wzWJgepoOossK6qanGVnCBY27hIMZ2IQ__Autumn_2020",),
    33: ("iIvOTBBXaRQmpWkFTFC9WLRoeNoDSLA8__Autumn_2020",),
    34: ("YEDRWy1MYuf5IONz4gQmQwAVuVQzkovm__Autumn_2020",),
    35: (
        "9SHZcSSJ6WhZgxrAg6eL7y9qvpPrpJA6__Autumn_2020",
        "9139e0ab9af848aabc3099e4f022db72__Autumn_2020",
    ),
    36: ("D8bJQtULZioDwmjylm0i6DYiU8Iarna0__Autumn_2020",),
    37: (
        "ByofqyOoTAkzeIocgXS7046IVehE4nGW__Autumn_2020",
        "d7b3375238c941ca8f1bbc34bc276bb1__Autumn_2020",
    ),
    38: ("hhkdx3VMk8RMtc8FMeBuRKRm1mJWdQwY__Autumn_2020",),
    39: ("yqNHd1H76GMchv6m66GzyYnPRNCpCDcT__Autumn_2020",),
    40: ("iwTFTnqVb6WjR5iw8tjRgCN28zaBKgAY__Autumn_2020",),
    41: ("tloQ89jLQcw2FX4S1Xrg0WJrnJVGXlDO__Autumn_2020",),
    42: (
        "EhENzsWbQDmmxWbbGJXTPmeOcUDI3D0p__Autumn_2020",
        "zlfTcv8ae4rVh4rDM8wOsUoSglTbC5AO__Autumn_2020",
    ),
    43: ("FfRsLioLHgbMgCLGKs8YAIHrExaYp3YP__Summer_2020",),
    44: ("oxtpIablm3xjNLGDEsFA0to1xzsptvZJ__Autumn_2020",),
    45: ("Vt5DOhIJC86BMPS5PKd7zg4Mk8ipMjXA__Winter_2020",),
    46: ("p2Ukdm04pz5vFlpeQ69kdQpwiOXD0eAz__Summer_2020",),
    47: ("e2Vvd8qpD57RA2ewZ40Sl1lwzvALFXfu__Summer_2020",),
    48: ("zQ06g8U26AU3pRHhzNxLPm340nOybscg__Spring_2020",),
    49: ("WY0cVNmhg7LtAs5Eny78Csltv2tbjdsd__Winter_2021",),
    50: ("01bb304d7bd835f8bbef7086b688e35e__Summer_2019",),
}

DTW_CLUSTERS: Final[Dict[int, Tuple[str, ...]]] = {
    1: ("XMbQ7jb2aVss95K5Ie9wTHK3LUTX8nz0__Spring_2020",),
    2: ("9U3zmqrZK6BVF6yUmseXSe6P4U1h6jUZ__Summer_2020",),
    3: (
        "ps87cXMi82mKmZ66R5iUVqLOcE9ZdYps__Summer_2020",
        "ouQqWBO1kAyPrbkIx7LcBVQgInxdntTq__Summer_2020",
    ),
    4: (
        "lBEMpTsvTu1k0SjZYLeZdVVcxgyZO6Gz__Summer_2020",
        "gsYZ0RMDMFZdRo1vmdW2Hp0Pn6hnKoIw__Autumn_2020",
        "hAsGqwlkaBv1zyxXl2tE7uqg4epa36ir__Winter_2021",
    ),
    5: (
        "5irmKvMAXQjJjUCC4VFT1FC7qsI8a4Kg__Summer_2020",
        "vyWiJxFSWgVvk5Yd3KVeyGKauOLypJnC__Summer_2020",
        "HGDGffqH58I2WAoFYKuTr1OYts3JWk4j__Summer_2020",
        "PRRmJxg1eSDS95RprHX0M1Ord3OCMcsg__Summer_2020",
        "s9aRUw1JDzPZPRmcx2h0wQCO2cerF66I__Autumn_2020",
        "b86c6b2e0ed64f4f89f43c1b00391731__Autumn_2020",
        "gu2UwLc2nU5YgX6tzcmfbqpPXWTngAOt__Autumn_2020",
        "42d1b09f89cd4e1d866eff93fd05b80f__Autumn_2020",
        "DKJk1QVisLulVf1jAvxijRTN33EBkgUE__Autumn_2020",
        "Dgs3UYZvrcojLDw0QLWmD24dPn0jAXhC__Autumn_2020",
    ),
    6: ("aEEUxFn6Gm6XeqV4rLA9esoNpYUhs2Ql__Summer_2020",),
    7: ("8pOtV4RsIeb0byPPCM6212tiiGNUvCyB__Summer_2020",),
    8: (
        "zRVrsPX6zQD7NKHIuBs4h5SgijdoNpKw__Autumn_2020",
        "3V9lgpdSQfTMXj3HjG2yt2gDbkcTK0v1__Winter_2021",
        "WwiugaY1H3aulOMXUCHbFTZKLDoqWuNP__Winter_2021",
        "cAstq5i3iTdHouG6HZh90LdA6BMbTW2A__Winter_2021",
    ),
    9: (
        "GJ2Kd0JBW2QQKEst8XefgaKKKOjb5RoX__Summer_2020",
        "5c2322b6817b4bdd8a23a43244e712c0__Summer_2020",
    ),
    10: (
        "FJVcWS3nQWkh9XU4TJzmCvaOXlByM947__Summer_2020",
        "XE3HUDWFcKhT1LcGp8GS8HacmMmok9dc__Autumn_2020",
    ),
    11: ("MVnyJgZdDbxqbQu9h9Q8kSjbziatTgxa__Autumn_2020",),
    12: ("rdWmCWPFNUuwwh9MBpEKzOEqJsmMuuyk__Winter_2021",),
    13: (
        "i47t2l9q6nJQNDFbD6iI0MrQ11yq4JNB__Autumn_2020",
        "2135a974fc804227843bd33514afbf5a__Autumn_2020",
    ),
    14: ("G6HCUIYJ0crBy1BbY5z5zx1OtCA8NZF0__Summer_2020",),
    15: ("75e8adad50a6324587265e612db3d165__Autumn_2019",),
    16: (
        "B1Ja9WqVlz5CAkSv2xsaiJK6KQq6egiT__Spring_2020",
        "79O52mmzE8VvjKQy2P1U9qywm2tneeXS__Autumn_2020",
        "sYzy2sKcm1qvahStIJf8h4Pl7HBEma7x__Autumn_2020",
        "r86GRA2V3jgD7pswTLq6NLJJBQqq2EMs__Autumn_2020",
    ),
}

PAO_CLUSTERS: Final[Dict[int, Tuple[str, ...]]] = {
    1: ("KH70lsYpeBe1I2TqKoD4V5AKJXfcCfpO__Summer_2020",),
    2: ("X3zP8oWJB4ivWYkbv6X9dw2mc1lRsceY__Summer_2020",),
    3: ("TOQaS7eCZB2ns552jjhmQoPs2MbdkgC6__Summer_2020",),
    4: (
        "Nr6t0auYyTEC42fJNIqhkaSasyGjfV6E__Autumn_2020",
        "jDONAwGVoVt9Ml4ZQW3FaTgDuhIlnAnt__Autumn_2020",
        "cJiXgBbopBsjrK8AE9GuHcuJCAGsHodO__Autumn_2020",
    ),
    5: ("453e5558636338e3bf9b42b5ba0a6f1d__Summer_2020",),
}

TBV_LOG_CLUSTERS: Final[Dict[str, Dict[int, Tuple[str, ...]]]] = {
    "ATX": ATX_CLUSTERS,
    "MIA": MIA_CLUSTERS,
    "PIT": PIT_CLUSTERS,
    "WDC": WDC_CLUSTERS,
    "PAO": PAO_CLUSTERS,
    "DTW": DTW_CLUSTERS,
}
