from .arena import Arena
from .table_arena import TableArena
from .empty_arena import EmptyArena
from .coffee_table_arena import CoffeeTableArena
from .living_room_arena import LivingRoomTableArena
from .study_arena import StudyTableArena
from .kitchen_arena import KitchenTableArena

AGENTVIEW_CONFIG = {
    "floor": {
            "camera_name":"agentview",
            "pos":[0.8965773716836134, 5.216182733499864e-07, 0.65],
            "quat":[
                0.6182166934013367,
                0.3432307541370392,
                0.3432314395904541,
                0.6182177066802979,
            ],
    },
    "main_table": {
        "camera_name":"agentview",
        "pos":[0.6586131746834771, 0.0, 1.6103500240372423],
        "quat":[
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    },
    "coffee_table": {
        "camera_name":"agentview",
        "pos":[0.6586131746834771, 0.0, 1.6103500240372423],
        "quat":[
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    },
    "kitchen_table": {
        "camera_name":"agentview",
        "pos":[0.6586131746834771, 0.0, 1.6103500240372423],
        "quat":[
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    },
    "living_room_table": {
        "camera_name":"agentview",
        "pos":[0.6065773716836134, 0.0, 0.96],
        "quat":[
            0.6182166934013367,
            0.3432307541370392,
            0.3432314395904541,
            0.6182177066802979,
        ],
    },
    "study_table": {
        "camera_name":"agentview",
        "pos":[0.4586131746834771, 0.0, 1.6103500240372423],
        "quat":[
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    }
}