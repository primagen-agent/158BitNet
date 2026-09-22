"""Exact C-tokenizer adapter for DG-015 fixtures, not a generation backend."""
from pathlib import Path

from c_tokenizer import CTokenizer
from native_memory_encoder import BACKBONE_SHA256, sha_file
from value_transport import TransportError


class NativeValueCodec:
    vocab = 73448

    def __init__(self, probe, gguf, expected_tokenizer_sha256):
        self.probe, self.gguf = Path(probe), Path(gguf)
        self.backbone_sha256 = BACKBONE_SHA256
        self.tokenizer_sha256 = expected_tokenizer_sha256
        self.verify_identity()
        self.tokenizer = CTokenizer(str(self.probe), str(self.gguf))
        try:
            self.bos_id = self.tokenizer.bos()
            self._blocked = {self.bos_id, self.tokenizer.eos()}
            pieces = self.tokenizer.decode_pieces(list(range(self.vocab)))
            self._blocked.update(i for i, p in enumerate(pieces) if p.startswith(b'<|') and p.endswith(b'|>'))
        except BaseException:
            self.close(); raise

    def verify_identity(self):
        if sha_file(self.probe) != self.tokenizer_sha256 or sha_file(self.gguf) != self.backbone_sha256:
            raise TransportError('C tokenizer/backbone identity changed')

    def encode_with_bos(self, text):
        # Probe protocol has reserved commands; none are valid test prefix texts.
        if text in ('BOS?', 'EOS?') or '\r' in text or '\0' in text:
            raise TransportError('unsupported tokenizer protocol text')
        if len(text.replace('\\', '\\\\').replace('\n', '\\n').encode()) > 65536:
            raise TransportError('tokenizer wire budget exceeded')
        return tuple(self.tokenizer.encode(text, True))

    def decode_pieces(self, ids):
        return tuple(self.tokenizer.decode_pieces(list(ids)))

    def forbidden(self, token):
        return token in self._blocked

    def close(self):
        p = self.tokenizer._proc
        p.terminate(); p.wait(); p.stdin.close(); p.stdout.close()
