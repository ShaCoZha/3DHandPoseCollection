"""Local replay server with byte-range support for seeking over SSH port forwarding."""
import argparse
import functools
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReplayHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def send_head(self):
        self.byte_range = None
        header = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not header or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
        try:
            if not match or not size:
                raise ValueError()
            first, last = match.groups()
            if not first:
                suffix = int(last)
                if suffix <= 0:
                    raise ValueError()
                start, end = max(0,size-suffix), size-1
            else:
                start = int(first)
                end = min(int(last),size-1) if last else size-1
            if start >= size or start > end:
                raise ValueError()
        except ValueError:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        source = path.open('rb')
        source.seek(start)
        self.byte_range = (start,end)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Last-Modified',self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return source

    def copyfile(self, source, outputfile):
        try:
            if self.byte_range is None:
                return super().copyfile(source,outputfile)
            remaining = self.byte_range[1]-self.byte_range[0]+1
            while remaining:
                data = source.read(min(256*1024,remaining))
                if not data:
                    break
                outputfile.write(data)
                remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser cancelled a previous seek.


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    handler=functools.partial(ReplayHandler,directory=str(args.directory.resolve()))
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
        print(f'Replay: http://127.0.0.1:{args.port}/viewer.html',flush=True)
        server.serve_forever()
