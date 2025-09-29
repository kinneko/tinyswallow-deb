#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinySwallow HTTP Tokenizer (ModuleLLM tokenizer_type=2向け)
- ローカルフォルダ or HFリポジトリからAutoTokenizerをロード
- エンドポイント:
  GET  /              -> {"status":"ok","model":"..."}
  GET  /bos_id        -> {"bos_id": int|-1}
  GET  /eos_id        -> {"eos_id": int|-1}
  POST /encode        -> 入力: {"text":"..."} または {"messages":[...], "add_generation_prompt":bool}
                         出力: {"token_ids":[...], "len":N}
  POST /decode        -> 入力: {"token_ids":[...]} 出力: {"text":"..."}
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoTokenizer
import json
import argparse
import os
import sys

def load_tokenizer(model_id_or_dir: str):
    # ローカルディレクトリが存在すれば優先
    if os.path.isdir(model_id_or_dir):
        return AutoTokenizer.from_pretrained(model_id_or_dir)
    # HF上のIDとしてロード（要ネット/トークン）
    return AutoTokenizer.from_pretrained(model_id_or_dir)

class TokenizerHttp:
    def __init__(self, model_id_or_dir: str):
        self.tok = load_tokenizer(model_id_or_dir)

    def encode_from_text(self, text: str, system_content: str = "", add_generation_prompt: bool = True):
        # system+userのmessagesでテンプレ適用
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": text})
        templ = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.tok.encode(templ)

    def encode_from_messages(self, messages, add_generation_prompt: bool = True):
        templ = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.tok.encode(templ)

    def decode(self, token_ids):
        return self.tok.decode(token_ids)

    @property
    def bos_id(self):
        return self.tok.bos_token_id

    @property
    def eos_id(self):
        return self.tok.eos_token_id

class Request(BaseHTTPRequestHandler):
    server_version = 'TinySwallowTokenizer/1.0'
    timeout = 5

    def _write_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "":
            # / と同義に扱う
            path = "/"
        if path == "/":
            return self._write_json({"status":"ok","model":self.server.ctx["model_disp"]})
        elif path == "/bos_id":
            bid = self.server.ctx["tok"].bos_id
            return self._write_json({"bos_id": (-1 if bid is None else bid)})
        elif path == "/eos_id":
            eid = self.server.ctx["tok"].eos_id
            return self._write_json({"eos_id": (-1 if eid is None else eid)})
        else:
            return self._write_json({"error":"not found"}, 404)

    def do_POST(self):
        try:
            ln = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(ln)
            req = json.loads(body.decode("utf-8")) if ln > 0 else {}
        except Exception as e:
            return self._write_json({"error": f"bad json: {e}"}, 400)

        path = self.path

        if path == "/encode":
            try:
                add_gen = bool(req.get("add_generation_prompt", True))
                if "messages" in req and isinstance(req["messages"], list):
                    ids = self.server.ctx["tok"].encode_from_messages(req["messages"], add_generation_prompt=add_gen)
                else:
                    text = req.get("text", "")
                    # system が来ていれば優先。無ければ既定（--content）
                    system_content = req.get("system", self.server.ctx["default_system"])
                    ids = self.server.ctx["tok"].encode_from_text(text, system_content, add_generation_prompt=add_gen)
                return self._write_json({"token_ids": ids, "len": len(ids)})
            except Exception as e:
                return self._write_json({"error": f"encode failed: {e}"}, 500)

        elif path == "/decode":
            try:
                ids = req.get("token_ids", [])
                text = self.server.ctx["tok"].decode(ids)
                return self._write_json({"text": text})
            except Exception as e:
                return self._write_json({"error": f"decode failed: {e}"}, 500)

        else:
            return self._write_json({"error":"not found"}, 404)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', type=str, default=os.environ.get("TS_HOST","0.0.0.0"))
    ap.add_argument('--port', type=int, default=int(os.environ.get("TS_PORT","8080")))
    # TinySwallowのローカル配置ディレクトリを既定に
    ap.add_argument('--model_id', type=str,
                    default=os.environ.get("TS_MODEL_DIR",
                        "/opt/m5stack/data/tinyswallow-1.5b-ax630c"))
    ap.add_argument('--content', type=str, default=os.environ.get(
        "TS_SYSTEM",
        "You are TinySwallow, a helpful Japanese assistant. Reply concisely."
    ))
    args = ap.parse_args()

    try:
        tok = TokenizerHttp(args.model_id)
    except Exception as e:
        print(f"[FATAL] tokenizer load failed: {e}", file=sys.stderr)
        sys.exit(2)

    srv = HTTPServer((args.host, args.port), Request)
    # 共有コンテキスト
    srv.ctx = {
        "tok": tok,
        "default_system": args.content,
        "model_disp": args.model_id
    }
    print(f"http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
