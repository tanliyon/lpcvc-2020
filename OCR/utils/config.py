from yacs.config import CfgNode as CN
_C = CN()

_C.ALPHABETS = " :a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:A:B:C:D:E:F:G:H:I:J:K:L:M:N:O:P:Q:R:S:T:U:V:W:X:Y:Z:0:1:2:3:4:5:6:7:8:9"
# _C.ALPHABETS = " :a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:A:B:C:D:E:F:G:\
# H:I:J:K:L:M:N:O:P:Q:R:S:T:U:V:W:X:Y:Z:0:1:2:3:4:5:6:7:8:9:!:#:$:%:&:':\
# (:):*:+:,: :-:.:/:;:<:=:>:?:@:[:\:]:^:_:`:{:|:}:~:"
# _C.ALPHABETS = " :a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:0:1:2:3:4:5:6:7:8:9"
# _C.ALPHABETS = 'abcefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$'
_C.BATCH_SIZE = 64
# _C.TOTAL_CHAR = 63
_C.TOTAL_CHAR = 37
_C.NUM_EPOCH = 50
CONFIG = _C
