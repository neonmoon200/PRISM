import logging
import sys


def _init_logger() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # add stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    # 仿真时间≠墙钟：LLM 走真实 HTTP；httpx 默认 INFO 会刷屏且误导为“卡住”。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info("init logging")


_init_logger()
