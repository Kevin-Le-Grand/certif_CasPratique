й┴
░ 
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.11.02v2.11.0-rc2-15-g6290819256d8­З
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:-*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:-*
dtype0
ё
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђў-*$
shared_nameAdam/v/dense/kernel
}
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel* 
_output_shapes
:
ђў-*
dtype0
ё
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђў-*$
shared_nameAdam/m/dense/kernel
}
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel* 
_output_shapes
:
ђў-*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:-*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђў-*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
ђў-*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:ђ*
dtype0
ї
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock5_conv3/kernel
Ё
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:ђђ*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:ђ*
dtype0
ї
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock5_conv2/kernel
Ё
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:ђђ*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:ђ*
dtype0
ї
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock5_conv1/kernel
Ё
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:ђђ*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:ђ*
dtype0
ї
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock4_conv3/kernel
Ё
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:ђђ*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:ђ*
dtype0
ї
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock4_conv2/kernel
Ё
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:ђђ*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:ђ*
dtype0
ї
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock4_conv1/kernel
Ё
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:ђђ*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:ђ*
dtype0
ї
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock3_conv3/kernel
Ё
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:ђђ*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:ђ*
dtype0
ї
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock3_conv2/kernel
Ё
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:ђђ*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:ђ*
dtype0
ї
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock3_conv1/kernel
Ё
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:ђђ*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:ђ*
dtype0
ї
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*$
shared_nameblock2_conv2/kernel
Ё
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:ђђ*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:ђ*
dtype0
І
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*$
shared_nameblock2_conv1/kernel
ё
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@ђ*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
і
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
Ѓ
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
і
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
Ѓ
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
ј
serving_default_input_1Placeholder*1
_output_shapes
:         ­└*
dtype0*&
shape:         ­└
і
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense/kernel
dense/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference_signature_wrapper_3465

NoOpNoOp
Жі
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ці
valueЎіBЋі BЇі
п
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer-19
layer_with_weights-13
layer-20
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

_init_input_shape* 
╚
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op*
╚
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op*
ј
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
╚
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op*
╚
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias
 I_jit_compiled_convolution_op*
ј
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
╚
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias
 X_jit_compiled_convolution_op*
╚
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

_kernel
`bias
 a_jit_compiled_convolution_op*
╚
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op*
ј
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses* 
╚
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias
 y_jit_compiled_convolution_op*
╦
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
ђkernel
	Ђbias
!ѓ_jit_compiled_convolution_op*
Л
Ѓ	variables
ёtrainable_variables
Ёregularization_losses
є	keras_api
Є__call__
+ѕ&call_and_return_all_conditional_losses
Ѕkernel
	іbias
!І_jit_compiled_convolution_op*
ћ
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses* 
Л
њ	variables
Њtrainable_variables
ћregularization_losses
Ћ	keras_api
ќ__call__
+Ќ&call_and_return_all_conditional_losses
ўkernel
	Ўbias
!џ_jit_compiled_convolution_op*
Л
Џ	variables
юtrainable_variables
Юregularization_losses
ъ	keras_api
Ъ__call__
+а&call_and_return_all_conditional_losses
Аkernel
	бbias
!Б_jit_compiled_convolution_op*
Л
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
е__call__
+Е&call_and_return_all_conditional_losses
фkernel
	Фbias
!г_jit_compiled_convolution_op*
ћ
Г	variables
«trainable_variables
»regularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses* 
ћ
│	variables
┤trainable_variables
хregularization_losses
Х	keras_api
и__call__
+И&call_and_return_all_conditional_losses* 
«
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
й__call__
+Й&call_and_return_all_conditional_losses
┐kernel
	└bias*
Т
&0
'1
/2
03
>4
?5
G6
H7
V8
W9
_10
`11
h12
i13
w14
x15
ђ16
Ђ17
Ѕ18
і19
ў20
Ў21
А22
б23
ф24
Ф25
┐26
└27*

┐0
└1*
* 
х
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
кtrace_0
Кtrace_1
╚trace_2
╔trace_3* 
:
╩trace_0
╦trace_1
╠trace_2
═trace_3* 
* 
ѕ
╬
_variables
¤_iterations
л_learning_rate
Л_index_dict
м
_momentums
М_velocities
н_update_step_xla*

Нserving_default* 
* 

&0
'1*
* 
* 
ў
оnon_trainable_variables
Оlayers
пmetrics
 ┘layer_regularization_losses
┌layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

█trace_0* 

▄trace_0* 
c]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

/0
01*
* 
* 
ў
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Рtrace_0* 

сtrace_0* 
c]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
Сnon_trainable_variables
тlayers
Тmetrics
 уlayer_regularization_losses
Уlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

жtrace_0* 

Жtrace_0* 

>0
?1*
* 
* 
ў
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

­trace_0* 

ыtrace_0* 
c]
VARIABLE_VALUEblock2_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock2_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

G0
H1*
* 
* 
ў
Ыnon_trainable_variables
зlayers
Зmetrics
 шlayer_regularization_losses
Шlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

эtrace_0* 

Эtrace_0* 
c]
VARIABLE_VALUEblock2_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock2_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
щnon_trainable_variables
Щlayers
чmetrics
 Чlayer_regularization_losses
§layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

■trace_0* 

 trace_0* 

V0
W1*
* 
* 
ў
ђnon_trainable_variables
Ђlayers
ѓmetrics
 Ѓlayer_regularization_losses
ёlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

Ёtrace_0* 

єtrace_0* 
c]
VARIABLE_VALUEblock3_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock3_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

_0
`1*
* 
* 
ў
Єnon_trainable_variables
ѕlayers
Ѕmetrics
 іlayer_regularization_losses
Іlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

їtrace_0* 

Їtrace_0* 
c]
VARIABLE_VALUEblock3_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock3_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

h0
i1*
* 
* 
ў
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

Њtrace_0* 

ћtrace_0* 
c]
VARIABLE_VALUEblock3_conv3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock3_conv3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
Ћnon_trainable_variables
ќlayers
Ќmetrics
 ўlayer_regularization_losses
Ўlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 

џtrace_0* 

Џtrace_0* 

w0
x1*
* 
* 
ў
юnon_trainable_variables
Юlayers
ъmetrics
 Ъlayer_regularization_losses
аlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

Аtrace_0* 

бtrace_0* 
c]
VARIABLE_VALUEblock4_conv1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock4_conv1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

ђ0
Ђ1*
* 
* 
ў
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

еtrace_0* 

Еtrace_0* 
c]
VARIABLE_VALUEblock4_conv2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock4_conv2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ѕ0
і1*
* 
* 
ъ
фnon_trainable_variables
Фlayers
гmetrics
 Гlayer_regularization_losses
«layer_metrics
Ѓ	variables
ёtrainable_variables
Ёregularization_losses
Є__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses*

»trace_0* 

░trace_0* 
c]
VARIABLE_VALUEblock4_conv3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock4_conv3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
хlayer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses* 

Хtrace_0* 

иtrace_0* 

ў0
Ў1*
* 
* 
ъ
Иnon_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
њ	variables
Њtrainable_variables
ћregularization_losses
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses*

йtrace_0* 

Йtrace_0* 
d^
VARIABLE_VALUEblock5_conv1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock5_conv1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

А0
б1*
* 
* 
ъ
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
Џ	variables
юtrainable_variables
Юregularization_losses
Ъ__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses*

─trace_0* 

┼trace_0* 
d^
VARIABLE_VALUEblock5_conv2/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock5_conv2/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

ф0
Ф1*
* 
* 
ъ
кnon_trainable_variables
Кlayers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
ц	variables
Цtrainable_variables
дregularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses*

╦trace_0* 

╠trace_0* 
d^
VARIABLE_VALUEblock5_conv3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock5_conv3/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ю
═non_trainable_variables
╬layers
¤metrics
 лlayer_regularization_losses
Лlayer_metrics
Г	variables
«trainable_variables
»regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses* 

мtrace_0* 

Мtrace_0* 
* 
* 
* 
ю
нnon_trainable_variables
Нlayers
оmetrics
 Оlayer_regularization_losses
пlayer_metrics
│	variables
┤trainable_variables
хregularization_losses
и__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses* 

┘trace_0* 

┌trace_0* 

┐0
└1*

┐0
└1*
* 
ъ
█non_trainable_variables
▄layers
Пmetrics
 яlayer_regularization_losses
▀layer_metrics
╣	variables
║trainable_variables
╗regularization_losses
й__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses*

Яtrace_0* 

рtrace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
н
&0
'1
/2
03
>4
?5
G6
H7
V8
W9
_10
`11
h12
i13
w14
x15
ђ16
Ђ17
Ѕ18
і19
ў20
Ў21
А22
б23
ф24
Ф25*
б
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20*

Р0
с1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
,
¤0
С1
т2
Т3
у4*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

С0
Т1*

т0
у1*
* 
* 

&0
'1*
* 
* 
* 
* 
* 
* 

/0
01*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

>0
?1*
* 
* 
* 
* 
* 
* 

G0
H1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

V0
W1*
* 
* 
* 
* 
* 
* 

_0
`1*
* 
* 
* 
* 
* 
* 

h0
i1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

w0
x1*
* 
* 
* 
* 
* 
* 

ђ0
Ђ1*
* 
* 
* 
* 
* 
* 

Ѕ0
і1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ў0
Ў1*
* 
* 
* 
* 
* 
* 

А0
б1*
* 
* 
* 
* 
* 
* 

ф0
Ф1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
У	variables
ж	keras_api

Жtotal

вcount*
M
В	variables
ь	keras_api

Ьtotal

№count
­
_fn_kwargs*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*

Ж0
в1*

У	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ь0
№1*

В	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
с
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *&
f!R
__inference__traced_save_4283
Т
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense/kernel
dense/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biastotal_1count_1totalcount*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__traced_restore_4407шЂ
Ї
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_2510

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
г
F
*__inference_block1_pool_layer_call_fn_3850

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_2462Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ї
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_3905

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
г
F
*__inference_block4_pool_layer_call_fn_4040

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_2498Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Њ
Ђ
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2566

inputs9
conv2d_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         xађk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         xађw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         xа@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         xа@
 
_user_specified_nameinputs
Ќ
ѓ
F__inference_block2_conv2_layer_call_and_return_conditional_losses_3895

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         xађk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         xађw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         xађ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         xађ
 
_user_specified_nameinputs
Ї
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_3975

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш
Б
+__inference_block4_conv3_layer_call_fn_4024

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2687x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         (ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
К
]
A__inference_flatten_layer_call_and_return_conditional_losses_2752

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     ї  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђўZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђў"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         
ђ:X T
0
_output_shapes
:         
ђ
 
_user_specified_nameinputs
ь\
╩
?__inference_model_layer_call_and_return_conditional_losses_3320
input_1+
block1_conv1_3243:@
block1_conv1_3245:@+
block1_conv2_3248:@@
block1_conv2_3250:@,
block2_conv1_3254:@ђ 
block2_conv1_3256:	ђ-
block2_conv2_3259:ђђ 
block2_conv2_3261:	ђ-
block3_conv1_3265:ђђ 
block3_conv1_3267:	ђ-
block3_conv2_3270:ђђ 
block3_conv2_3272:	ђ-
block3_conv3_3275:ђђ 
block3_conv3_3277:	ђ-
block4_conv1_3281:ђђ 
block4_conv1_3283:	ђ-
block4_conv2_3286:ђђ 
block4_conv2_3288:	ђ-
block4_conv3_3291:ђђ 
block4_conv3_3293:	ђ-
block5_conv1_3297:ђђ 
block5_conv1_3299:	ђ-
block5_conv2_3302:ђђ 
block5_conv2_3304:	ђ-
block5_conv3_3307:ђђ 
block5_conv3_3309:	ђ

dense_3314:
ђў-

dense_3316:-
identityѕб$block1_conv1/StatefulPartitionedCallб$block1_conv2/StatefulPartitionedCallб$block2_conv1/StatefulPartitionedCallб$block2_conv2/StatefulPartitionedCallб$block3_conv1/StatefulPartitionedCallб$block3_conv2/StatefulPartitionedCallб$block3_conv3/StatefulPartitionedCallб$block4_conv1/StatefulPartitionedCallб$block4_conv2/StatefulPartitionedCallб$block4_conv3/StatefulPartitionedCallб$block5_conv1/StatefulPartitionedCallб$block5_conv2/StatefulPartitionedCallб$block5_conv3/StatefulPartitionedCallбdense/StatefulPartitionedCallЁ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_3243block1_conv1_3245*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2531Ф
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_3248block1_conv2_3250*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2548В
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xа@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_2462б
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_3254block2_conv1_3256*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2566Ф
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_3259block2_conv2_3261*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2583В
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_2474А
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_3265block3_conv1_3267*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2601ф
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_3270block3_conv2_3272*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2618ф
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_3275block3_conv3_3277*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2635В
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_2486А
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_3281block4_conv1_3283*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2653ф
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_3286block4_conv2_3288*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2670ф
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_3291block4_conv3_3293*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2687В
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_2498А
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_3297block5_conv1_3299*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2705ф
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_3302block5_conv2_3304*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2722ф
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_3307block5_conv3_3309*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2739В
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         
ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_2510н
flatten/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_2752Э
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_3314
dense_3316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2765u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -р
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:         ­└
!
_user_specified_name	input_1
Ш
б
+__inference_block2_conv1_layer_call_fn_3864

inputs"
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2566y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         xађ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         xа@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         xа@
 
_user_specified_nameinputs
г
F
*__inference_block5_pool_layer_call_fn_4110

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_2510Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ж\
╔
?__inference_model_layer_call_and_return_conditional_losses_2772

inputs+
block1_conv1_2532:@
block1_conv1_2534:@+
block1_conv2_2549:@@
block1_conv2_2551:@,
block2_conv1_2567:@ђ 
block2_conv1_2569:	ђ-
block2_conv2_2584:ђђ 
block2_conv2_2586:	ђ-
block3_conv1_2602:ђђ 
block3_conv1_2604:	ђ-
block3_conv2_2619:ђђ 
block3_conv2_2621:	ђ-
block3_conv3_2636:ђђ 
block3_conv3_2638:	ђ-
block4_conv1_2654:ђђ 
block4_conv1_2656:	ђ-
block4_conv2_2671:ђђ 
block4_conv2_2673:	ђ-
block4_conv3_2688:ђђ 
block4_conv3_2690:	ђ-
block5_conv1_2706:ђђ 
block5_conv1_2708:	ђ-
block5_conv2_2723:ђђ 
block5_conv2_2725:	ђ-
block5_conv3_2740:ђђ 
block5_conv3_2742:	ђ

dense_2766:
ђў-

dense_2768:-
identityѕб$block1_conv1/StatefulPartitionedCallб$block1_conv2/StatefulPartitionedCallб$block2_conv1/StatefulPartitionedCallб$block2_conv2/StatefulPartitionedCallб$block3_conv1/StatefulPartitionedCallб$block3_conv2/StatefulPartitionedCallб$block3_conv3/StatefulPartitionedCallб$block4_conv1/StatefulPartitionedCallб$block4_conv2/StatefulPartitionedCallб$block4_conv3/StatefulPartitionedCallб$block5_conv1/StatefulPartitionedCallб$block5_conv2/StatefulPartitionedCallб$block5_conv3/StatefulPartitionedCallбdense/StatefulPartitionedCallё
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_2532block1_conv1_2534*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2531Ф
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_2549block1_conv2_2551*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2548В
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xа@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_2462б
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_2567block2_conv1_2569*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2566Ф
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_2584block2_conv2_2586*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2583В
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_2474А
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_2602block3_conv1_2604*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2601ф
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_2619block3_conv2_2621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2618ф
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_2636block3_conv3_2638*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2635В
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_2486А
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_2654block4_conv1_2656*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2653ф
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_2671block4_conv2_2673*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2670ф
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_2688block4_conv3_2690*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2687В
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_2498А
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_2706block5_conv1_2708*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2705ф
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_2723block5_conv2_2725*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2722ф
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_2740block5_conv3_2742*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2739В
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         
ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_2510н
flatten/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_2752Э
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_2766
dense_2768*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2765u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -р
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
зљ
а
?__inference_model_layer_call_and_return_conditional_losses_3805

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@ђ;
,block2_conv1_biasadd_readvariableop_resource:	ђG
+block2_conv2_conv2d_readvariableop_resource:ђђ;
,block2_conv2_biasadd_readvariableop_resource:	ђG
+block3_conv1_conv2d_readvariableop_resource:ђђ;
,block3_conv1_biasadd_readvariableop_resource:	ђG
+block3_conv2_conv2d_readvariableop_resource:ђђ;
,block3_conv2_biasadd_readvariableop_resource:	ђG
+block3_conv3_conv2d_readvariableop_resource:ђђ;
,block3_conv3_biasadd_readvariableop_resource:	ђG
+block4_conv1_conv2d_readvariableop_resource:ђђ;
,block4_conv1_biasadd_readvariableop_resource:	ђG
+block4_conv2_conv2d_readvariableop_resource:ђђ;
,block4_conv2_biasadd_readvariableop_resource:	ђG
+block4_conv3_conv2d_readvariableop_resource:ђђ;
,block4_conv3_biasadd_readvariableop_resource:	ђG
+block5_conv1_conv2d_readvariableop_resource:ђђ;
,block5_conv1_biasadd_readvariableop_resource:	ђG
+block5_conv2_conv2d_readvariableop_resource:ђђ;
,block5_conv2_biasadd_readvariableop_resource:	ђG
+block5_conv3_conv2d_readvariableop_resource:ђђ;
,block5_conv3_biasadd_readvariableop_resource:	ђ8
$dense_matmul_readvariableop_resource:
ђў-3
%dense_biasadd_readvariableop_resource:-
identityѕб#block1_conv1/BiasAdd/ReadVariableOpб"block1_conv1/Conv2D/ReadVariableOpб#block1_conv2/BiasAdd/ReadVariableOpб"block1_conv2/Conv2D/ReadVariableOpб#block2_conv1/BiasAdd/ReadVariableOpб"block2_conv1/Conv2D/ReadVariableOpб#block2_conv2/BiasAdd/ReadVariableOpб"block2_conv2/Conv2D/ReadVariableOpб#block3_conv1/BiasAdd/ReadVariableOpб"block3_conv1/Conv2D/ReadVariableOpб#block3_conv2/BiasAdd/ReadVariableOpб"block3_conv2/Conv2D/ReadVariableOpб#block3_conv3/BiasAdd/ReadVariableOpб"block3_conv3/Conv2D/ReadVariableOpб#block4_conv1/BiasAdd/ReadVariableOpб"block4_conv1/Conv2D/ReadVariableOpб#block4_conv2/BiasAdd/ReadVariableOpб"block4_conv2/Conv2D/ReadVariableOpб#block4_conv3/BiasAdd/ReadVariableOpб"block4_conv3/Conv2D/ReadVariableOpб#block5_conv1/BiasAdd/ReadVariableOpб"block5_conv1/Conv2D/ReadVariableOpб#block5_conv2/BiasAdd/ReadVariableOpб"block5_conv2/Conv2D/ReadVariableOpб#block5_conv3/BiasAdd/ReadVariableOpб"block5_conv3/Conv2D/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpќ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0х
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ­└@ќ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╬
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         ­└@Г
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*0
_output_shapes
:         xа@*
ksize
*
paddingVALID*
strides
Ќ
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0╦
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
Ї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0д
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађt
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         xађў
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╬
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
Ї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0д
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађt
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         xађГ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:         <Pђ*
ksize
*
paddingVALID*
strides
ў
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╩
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         <Pђў
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         <Pђў
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ї
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђs
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         <PђГ
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:         (ђ*
ksize
*
paddingVALID*
strides
ў
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╩
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ї
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђў
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ї
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђў
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ї
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђГ
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
ў
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╩
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ї
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         ђў
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ї
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         ђў
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ї
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         ђГ
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:         
ђ*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     ї  ё
flatten/ReshapeReshapeblock5_pool/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:         ђўѓ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђў-*
dtype0Є
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:         -f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         -м
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
Ї
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_2462

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш
Б
+__inference_block3_conv3_layer_call_fn_3954

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2635x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         <Pђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
щ
Б
+__inference_block2_conv2_layer_call_fn_3884

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2583y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         xађ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         xађ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         xађ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block4_conv3_layer_call_and_return_conditional_losses_4035

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         (ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         (ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block3_conv1_layer_call_and_return_conditional_losses_3925

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <PђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         <Pђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         <Pђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
г
F
*__inference_block3_pool_layer_call_fn_3970

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_2486Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Љ
 
F__inference_block1_conv2_layer_call_and_return_conditional_losses_3845

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ­└@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ­└@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ­└@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ­└@
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block5_conv2_layer_call_and_return_conditional_losses_4085

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ї
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_4045

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block3_conv3_layer_call_and_return_conditional_losses_3965

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <PђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         <Pђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         <Pђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
Ш
а
+__inference_block1_conv2_layer_call_fn_3834

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2548y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ­└@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ­└@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ­└@
 
_user_specified_nameinputs
Љ
 
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2531

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ­└@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ­└@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ­└: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
Ї
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_4115

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
г
F
*__inference_block2_pool_layer_call_fn_3900

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_2474Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ж\
╔
?__inference_model_layer_call_and_return_conditional_losses_3120

inputs+
block1_conv1_3043:@
block1_conv1_3045:@+
block1_conv2_3048:@@
block1_conv2_3050:@,
block2_conv1_3054:@ђ 
block2_conv1_3056:	ђ-
block2_conv2_3059:ђђ 
block2_conv2_3061:	ђ-
block3_conv1_3065:ђђ 
block3_conv1_3067:	ђ-
block3_conv2_3070:ђђ 
block3_conv2_3072:	ђ-
block3_conv3_3075:ђђ 
block3_conv3_3077:	ђ-
block4_conv1_3081:ђђ 
block4_conv1_3083:	ђ-
block4_conv2_3086:ђђ 
block4_conv2_3088:	ђ-
block4_conv3_3091:ђђ 
block4_conv3_3093:	ђ-
block5_conv1_3097:ђђ 
block5_conv1_3099:	ђ-
block5_conv2_3102:ђђ 
block5_conv2_3104:	ђ-
block5_conv3_3107:ђђ 
block5_conv3_3109:	ђ

dense_3114:
ђў-

dense_3116:-
identityѕб$block1_conv1/StatefulPartitionedCallб$block1_conv2/StatefulPartitionedCallб$block2_conv1/StatefulPartitionedCallб$block2_conv2/StatefulPartitionedCallб$block3_conv1/StatefulPartitionedCallб$block3_conv2/StatefulPartitionedCallб$block3_conv3/StatefulPartitionedCallб$block4_conv1/StatefulPartitionedCallб$block4_conv2/StatefulPartitionedCallб$block4_conv3/StatefulPartitionedCallб$block5_conv1/StatefulPartitionedCallб$block5_conv2/StatefulPartitionedCallб$block5_conv3/StatefulPartitionedCallбdense/StatefulPartitionedCallё
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_3043block1_conv1_3045*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2531Ф
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_3048block1_conv2_3050*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2548В
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xа@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_2462б
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_3054block2_conv1_3056*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2566Ф
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_3059block2_conv2_3061*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2583В
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_2474А
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_3065block3_conv1_3067*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2601ф
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_3070block3_conv2_3072*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2618ф
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_3075block3_conv3_3077*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2635В
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_2486А
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_3081block4_conv1_3083*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2653ф
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_3086block4_conv2_3088*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2670ф
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_3091block4_conv3_3093*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2687В
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_2498А
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_3097block5_conv1_3099*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2705ф
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_3102block5_conv2_3104*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2722ф
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_3107block5_conv3_3109*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2739В
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         
ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_2510н
flatten/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_2752Э
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_3114
dense_3116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2765u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -р
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
┐
в
$__inference_model_layer_call_fn_3240
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ&

unknown_13:ђђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ&

unknown_19:ђђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ&

unknown_23:ђђ

unknown_24:	ђ

unknown_25:
ђў-

unknown_26:-
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_3120o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ­└
!
_user_specified_name	input_1
Љ
ѓ
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2635

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <PђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         <Pђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         <Pђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block5_conv1_layer_call_and_return_conditional_losses_4065

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
ш
Б
+__inference_block5_conv2_layer_call_fn_4074

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2722x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2601

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <PђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         <Pђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         <Pђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
йM
░
__inference__traced_save_4283
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: А
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*╩
value└Bй'B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╗
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B г
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *5
dtypes+
)2'	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*│
_input_shapesА
ъ: :@:@:@@:@:@ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:
ђў-:-: : :
ђў-:
ђў-:-:-: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.	*
(
_output_shapes
:ђђ:!


_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђў-: 

_output_shapes
:-:

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
ђў-:& "
 
_output_shapes
:
ђў-: !

_output_shapes
:-: "

_output_shapes
:-:#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
ш
Б
+__inference_block5_conv1_layer_call_fn_4054

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2705x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block3_conv2_layer_call_and_return_conditional_losses_3945

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <PђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         <Pђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         <Pђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
ш
Б
+__inference_block3_conv1_layer_call_fn_3914

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2601x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         <Pђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block4_conv1_layer_call_and_return_conditional_losses_3995

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         (ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         (ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
ш
Б
+__inference_block3_conv2_layer_call_fn_3934

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2618x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         <Pђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
Ќ
ѓ
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2583

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         xађk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         xађw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         xађ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         xађ
 
_user_specified_nameinputs
уЪ
Л
__inference__wrapped_model_2453
input_1K
1model_block1_conv1_conv2d_readvariableop_resource:@@
2model_block1_conv1_biasadd_readvariableop_resource:@K
1model_block1_conv2_conv2d_readvariableop_resource:@@@
2model_block1_conv2_biasadd_readvariableop_resource:@L
1model_block2_conv1_conv2d_readvariableop_resource:@ђA
2model_block2_conv1_biasadd_readvariableop_resource:	ђM
1model_block2_conv2_conv2d_readvariableop_resource:ђђA
2model_block2_conv2_biasadd_readvariableop_resource:	ђM
1model_block3_conv1_conv2d_readvariableop_resource:ђђA
2model_block3_conv1_biasadd_readvariableop_resource:	ђM
1model_block3_conv2_conv2d_readvariableop_resource:ђђA
2model_block3_conv2_biasadd_readvariableop_resource:	ђM
1model_block3_conv3_conv2d_readvariableop_resource:ђђA
2model_block3_conv3_biasadd_readvariableop_resource:	ђM
1model_block4_conv1_conv2d_readvariableop_resource:ђђA
2model_block4_conv1_biasadd_readvariableop_resource:	ђM
1model_block4_conv2_conv2d_readvariableop_resource:ђђA
2model_block4_conv2_biasadd_readvariableop_resource:	ђM
1model_block4_conv3_conv2d_readvariableop_resource:ђђA
2model_block4_conv3_biasadd_readvariableop_resource:	ђM
1model_block5_conv1_conv2d_readvariableop_resource:ђђA
2model_block5_conv1_biasadd_readvariableop_resource:	ђM
1model_block5_conv2_conv2d_readvariableop_resource:ђђA
2model_block5_conv2_biasadd_readvariableop_resource:	ђM
1model_block5_conv3_conv2d_readvariableop_resource:ђђA
2model_block5_conv3_biasadd_readvariableop_resource:	ђ>
*model_dense_matmul_readvariableop_resource:
ђў-9
+model_dense_biasadd_readvariableop_resource:-
identityѕб)model/block1_conv1/BiasAdd/ReadVariableOpб(model/block1_conv1/Conv2D/ReadVariableOpб)model/block1_conv2/BiasAdd/ReadVariableOpб(model/block1_conv2/Conv2D/ReadVariableOpб)model/block2_conv1/BiasAdd/ReadVariableOpб(model/block2_conv1/Conv2D/ReadVariableOpб)model/block2_conv2/BiasAdd/ReadVariableOpб(model/block2_conv2/Conv2D/ReadVariableOpб)model/block3_conv1/BiasAdd/ReadVariableOpб(model/block3_conv1/Conv2D/ReadVariableOpб)model/block3_conv2/BiasAdd/ReadVariableOpб(model/block3_conv2/Conv2D/ReadVariableOpб)model/block3_conv3/BiasAdd/ReadVariableOpб(model/block3_conv3/Conv2D/ReadVariableOpб)model/block4_conv1/BiasAdd/ReadVariableOpб(model/block4_conv1/Conv2D/ReadVariableOpб)model/block4_conv2/BiasAdd/ReadVariableOpб(model/block4_conv2/Conv2D/ReadVariableOpб)model/block4_conv3/BiasAdd/ReadVariableOpб(model/block4_conv3/Conv2D/ReadVariableOpб)model/block5_conv1/BiasAdd/ReadVariableOpб(model/block5_conv1/Conv2D/ReadVariableOpб)model/block5_conv2/BiasAdd/ReadVariableOpб(model/block5_conv2/Conv2D/ReadVariableOpб)model/block5_conv3/BiasAdd/ReadVariableOpб(model/block5_conv3/Conv2D/ReadVariableOpб"model/dense/BiasAdd/ReadVariableOpб!model/dense/MatMul/ReadVariableOpб
(model/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0┬
model/block1_conv1/Conv2DConv2Dinput_10model/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
ў
)model/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0И
model/block1_conv1/BiasAddBiasAdd"model/block1_conv1/Conv2D:output:01model/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@ђ
model/block1_conv1/ReluRelu#model/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ­└@б
(model/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Я
model/block1_conv2/Conv2DConv2D%model/block1_conv1/Relu:activations:00model/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
ў
)model/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0И
model/block1_conv2/BiasAddBiasAdd"model/block1_conv2/Conv2D:output:01model/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@ђ
model/block1_conv2/ReluRelu#model/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         ­└@╣
model/block1_pool/MaxPoolMaxPool%model/block1_conv2/Relu:activations:0*0
_output_shapes
:         xа@*
ksize
*
paddingVALID*
strides
Б
(model/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0П
model/block2_conv1/Conv2DConv2D"model/block1_pool/MaxPool:output:00model/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
Ў
)model/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0И
model/block2_conv1/BiasAddBiasAdd"model/block2_conv1/Conv2D:output:01model/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађђ
model/block2_conv1/ReluRelu#model/block2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         xађц
(model/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Я
model/block2_conv2/Conv2DConv2D%model/block2_conv1/Relu:activations:00model/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
Ў
)model/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0И
model/block2_conv2/BiasAddBiasAdd"model/block2_conv2/Conv2D:output:01model/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађђ
model/block2_conv2/ReluRelu#model/block2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         xађ╣
model/block2_pool/MaxPoolMaxPool%model/block2_conv2/Relu:activations:0*0
_output_shapes
:         <Pђ*
ksize
*
paddingVALID*
strides
ц
(model/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▄
model/block3_conv1/Conv2DConv2D"model/block2_pool/MaxPool:output:00model/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ў
)model/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block3_conv1/BiasAddBiasAdd"model/block3_conv1/Conv2D:output:01model/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ
model/block3_conv1/ReluRelu#model/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         <Pђц
(model/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▀
model/block3_conv2/Conv2DConv2D%model/block3_conv1/Relu:activations:00model/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ў
)model/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block3_conv2/BiasAddBiasAdd"model/block3_conv2/Conv2D:output:01model/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ
model/block3_conv2/ReluRelu#model/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         <Pђц
(model/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▀
model/block3_conv3/Conv2DConv2D%model/block3_conv2/Relu:activations:00model/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ў
)model/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block3_conv3/BiasAddBiasAdd"model/block3_conv3/Conv2D:output:01model/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ
model/block3_conv3/ReluRelu#model/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         <Pђ╣
model/block3_pool/MaxPoolMaxPool%model/block3_conv3/Relu:activations:0*0
_output_shapes
:         (ђ*
ksize
*
paddingVALID*
strides
ц
(model/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▄
model/block4_conv1/Conv2DConv2D"model/block3_pool/MaxPool:output:00model/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ў
)model/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block4_conv1/BiasAddBiasAdd"model/block4_conv1/Conv2D:output:01model/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ
model/block4_conv1/ReluRelu#model/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђц
(model/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▀
model/block4_conv2/Conv2DConv2D%model/block4_conv1/Relu:activations:00model/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ў
)model/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block4_conv2/BiasAddBiasAdd"model/block4_conv2/Conv2D:output:01model/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ
model/block4_conv2/ReluRelu#model/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђц
(model/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▀
model/block4_conv3/Conv2DConv2D%model/block4_conv2/Relu:activations:00model/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ў
)model/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block4_conv3/BiasAddBiasAdd"model/block4_conv3/Conv2D:output:01model/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ
model/block4_conv3/ReluRelu#model/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђ╣
model/block4_pool/MaxPoolMaxPool%model/block4_conv3/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
ц
(model/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▄
model/block5_conv1/Conv2DConv2D"model/block4_pool/MaxPool:output:00model/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ў
)model/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block5_conv1/BiasAddBiasAdd"model/block5_conv1/Conv2D:output:01model/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ
model/block5_conv1/ReluRelu#model/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         ђц
(model/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▀
model/block5_conv2/Conv2DConv2D%model/block5_conv1/Relu:activations:00model/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ў
)model/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block5_conv2/BiasAddBiasAdd"model/block5_conv2/Conv2D:output:01model/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ
model/block5_conv2/ReluRelu#model/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         ђц
(model/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▀
model/block5_conv3/Conv2DConv2D%model/block5_conv2/Relu:activations:00model/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ў
)model/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
model/block5_conv3/BiasAddBiasAdd"model/block5_conv3/Conv2D:output:01model/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ
model/block5_conv3/ReluRelu#model/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ╣
model/block5_pool/MaxPoolMaxPool%model/block5_conv3/Relu:activations:0*0
_output_shapes
:         
ђ*
ksize
*
paddingVALID*
strides
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     ї  ќ
model/flatten/ReshapeReshape"model/block5_pool/MaxPool:output:0model/flatten/Const:output:0*
T0*)
_output_shapes
:         ђўј
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђў-*
dtype0Ў
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -і
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0џ
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -n
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         -l
IdentityIdentitymodel/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         -Щ	
NoOpNoOp*^model/block1_conv1/BiasAdd/ReadVariableOp)^model/block1_conv1/Conv2D/ReadVariableOp*^model/block1_conv2/BiasAdd/ReadVariableOp)^model/block1_conv2/Conv2D/ReadVariableOp*^model/block2_conv1/BiasAdd/ReadVariableOp)^model/block2_conv1/Conv2D/ReadVariableOp*^model/block2_conv2/BiasAdd/ReadVariableOp)^model/block2_conv2/Conv2D/ReadVariableOp*^model/block3_conv1/BiasAdd/ReadVariableOp)^model/block3_conv1/Conv2D/ReadVariableOp*^model/block3_conv2/BiasAdd/ReadVariableOp)^model/block3_conv2/Conv2D/ReadVariableOp*^model/block3_conv3/BiasAdd/ReadVariableOp)^model/block3_conv3/Conv2D/ReadVariableOp*^model/block4_conv1/BiasAdd/ReadVariableOp)^model/block4_conv1/Conv2D/ReadVariableOp*^model/block4_conv2/BiasAdd/ReadVariableOp)^model/block4_conv2/Conv2D/ReadVariableOp*^model/block4_conv3/BiasAdd/ReadVariableOp)^model/block4_conv3/Conv2D/ReadVariableOp*^model/block5_conv1/BiasAdd/ReadVariableOp)^model/block5_conv1/Conv2D/ReadVariableOp*^model/block5_conv2/BiasAdd/ReadVariableOp)^model/block5_conv2/Conv2D/ReadVariableOp*^model/block5_conv3/BiasAdd/ReadVariableOp)^model/block5_conv3/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model/block1_conv1/BiasAdd/ReadVariableOp)model/block1_conv1/BiasAdd/ReadVariableOp2T
(model/block1_conv1/Conv2D/ReadVariableOp(model/block1_conv1/Conv2D/ReadVariableOp2V
)model/block1_conv2/BiasAdd/ReadVariableOp)model/block1_conv2/BiasAdd/ReadVariableOp2T
(model/block1_conv2/Conv2D/ReadVariableOp(model/block1_conv2/Conv2D/ReadVariableOp2V
)model/block2_conv1/BiasAdd/ReadVariableOp)model/block2_conv1/BiasAdd/ReadVariableOp2T
(model/block2_conv1/Conv2D/ReadVariableOp(model/block2_conv1/Conv2D/ReadVariableOp2V
)model/block2_conv2/BiasAdd/ReadVariableOp)model/block2_conv2/BiasAdd/ReadVariableOp2T
(model/block2_conv2/Conv2D/ReadVariableOp(model/block2_conv2/Conv2D/ReadVariableOp2V
)model/block3_conv1/BiasAdd/ReadVariableOp)model/block3_conv1/BiasAdd/ReadVariableOp2T
(model/block3_conv1/Conv2D/ReadVariableOp(model/block3_conv1/Conv2D/ReadVariableOp2V
)model/block3_conv2/BiasAdd/ReadVariableOp)model/block3_conv2/BiasAdd/ReadVariableOp2T
(model/block3_conv2/Conv2D/ReadVariableOp(model/block3_conv2/Conv2D/ReadVariableOp2V
)model/block3_conv3/BiasAdd/ReadVariableOp)model/block3_conv3/BiasAdd/ReadVariableOp2T
(model/block3_conv3/Conv2D/ReadVariableOp(model/block3_conv3/Conv2D/ReadVariableOp2V
)model/block4_conv1/BiasAdd/ReadVariableOp)model/block4_conv1/BiasAdd/ReadVariableOp2T
(model/block4_conv1/Conv2D/ReadVariableOp(model/block4_conv1/Conv2D/ReadVariableOp2V
)model/block4_conv2/BiasAdd/ReadVariableOp)model/block4_conv2/BiasAdd/ReadVariableOp2T
(model/block4_conv2/Conv2D/ReadVariableOp(model/block4_conv2/Conv2D/ReadVariableOp2V
)model/block4_conv3/BiasAdd/ReadVariableOp)model/block4_conv3/BiasAdd/ReadVariableOp2T
(model/block4_conv3/Conv2D/ReadVariableOp(model/block4_conv3/Conv2D/ReadVariableOp2V
)model/block5_conv1/BiasAdd/ReadVariableOp)model/block5_conv1/BiasAdd/ReadVariableOp2T
(model/block5_conv1/Conv2D/ReadVariableOp(model/block5_conv1/Conv2D/ReadVariableOp2V
)model/block5_conv2/BiasAdd/ReadVariableOp)model/block5_conv2/BiasAdd/ReadVariableOp2T
(model/block5_conv2/Conv2D/ReadVariableOp(model/block5_conv2/Conv2D/ReadVariableOp2V
)model/block5_conv3/BiasAdd/ReadVariableOp)model/block5_conv3/BiasAdd/ReadVariableOp2T
(model/block5_conv3/Conv2D/ReadVariableOp(model/block5_conv3/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp:Z V
1
_output_shapes
:         ­└
!
_user_specified_name	input_1
Б

Ы
?__inference_dense_layer_call_and_return_conditional_losses_2765

inputs2
matmul_readvariableop_resource:
ђў--
biasadd_readvariableop_resource:-
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђў-*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         -`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         -w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђў: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         ђў
 
_user_specified_nameinputs
ш
Б
+__inference_block4_conv1_layer_call_fn_3984

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2653x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         (ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block5_conv3_layer_call_and_return_conditional_losses_4105

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2739

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2653

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         (ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         (ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
Љ
 
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2548

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ­└@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ­└@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ­└@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ­└@
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2687

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         (ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         (ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
ь\
╩
?__inference_model_layer_call_and_return_conditional_losses_3400
input_1+
block1_conv1_3323:@
block1_conv1_3325:@+
block1_conv2_3328:@@
block1_conv2_3330:@,
block2_conv1_3334:@ђ 
block2_conv1_3336:	ђ-
block2_conv2_3339:ђђ 
block2_conv2_3341:	ђ-
block3_conv1_3345:ђђ 
block3_conv1_3347:	ђ-
block3_conv2_3350:ђђ 
block3_conv2_3352:	ђ-
block3_conv3_3355:ђђ 
block3_conv3_3357:	ђ-
block4_conv1_3361:ђђ 
block4_conv1_3363:	ђ-
block4_conv2_3366:ђђ 
block4_conv2_3368:	ђ-
block4_conv3_3371:ђђ 
block4_conv3_3373:	ђ-
block5_conv1_3377:ђђ 
block5_conv1_3379:	ђ-
block5_conv2_3382:ђђ 
block5_conv2_3384:	ђ-
block5_conv3_3387:ђђ 
block5_conv3_3389:	ђ

dense_3394:
ђў-

dense_3396:-
identityѕб$block1_conv1/StatefulPartitionedCallб$block1_conv2/StatefulPartitionedCallб$block2_conv1/StatefulPartitionedCallб$block2_conv2/StatefulPartitionedCallб$block3_conv1/StatefulPartitionedCallб$block3_conv2/StatefulPartitionedCallб$block3_conv3/StatefulPartitionedCallб$block4_conv1/StatefulPartitionedCallб$block4_conv2/StatefulPartitionedCallб$block4_conv3/StatefulPartitionedCallб$block5_conv1/StatefulPartitionedCallб$block5_conv2/StatefulPartitionedCallб$block5_conv3/StatefulPartitionedCallбdense/StatefulPartitionedCallЁ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_3323block1_conv1_3325*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2531Ф
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_3328block1_conv2_3330*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2548В
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         xа@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_2462б
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_3334block2_conv1_3336*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2566Ф
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_3339block2_conv2_3341*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         xађ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2583В
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_2474А
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_3345block3_conv1_3347*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2601ф
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_3350block3_conv2_3352*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2618ф
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_3355block3_conv3_3357*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         <Pђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2635В
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_2486А
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_3361block4_conv1_3363*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2653ф
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_3366block4_conv2_3368*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2670ф
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_3371block4_conv3_3373*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2687В
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_2498А
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_3377block5_conv1_3379*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2705ф
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_3382block5_conv2_3384*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2722ф
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_3387block5_conv3_3389*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2739В
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         
ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_2510н
flatten/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_2752Э
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_3394
dense_3396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2765u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -р
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:         ­└
!
_user_specified_name	input_1
жа
р
 __inference__traced_restore_4407
file_prefix>
$assignvariableop_block1_conv1_kernel:@2
$assignvariableop_1_block1_conv1_bias:@@
&assignvariableop_2_block1_conv2_kernel:@@2
$assignvariableop_3_block1_conv2_bias:@A
&assignvariableop_4_block2_conv1_kernel:@ђ3
$assignvariableop_5_block2_conv1_bias:	ђB
&assignvariableop_6_block2_conv2_kernel:ђђ3
$assignvariableop_7_block2_conv2_bias:	ђB
&assignvariableop_8_block3_conv1_kernel:ђђ3
$assignvariableop_9_block3_conv1_bias:	ђC
'assignvariableop_10_block3_conv2_kernel:ђђ4
%assignvariableop_11_block3_conv2_bias:	ђC
'assignvariableop_12_block3_conv3_kernel:ђђ4
%assignvariableop_13_block3_conv3_bias:	ђC
'assignvariableop_14_block4_conv1_kernel:ђђ4
%assignvariableop_15_block4_conv1_bias:	ђC
'assignvariableop_16_block4_conv2_kernel:ђђ4
%assignvariableop_17_block4_conv2_bias:	ђC
'assignvariableop_18_block4_conv3_kernel:ђђ4
%assignvariableop_19_block4_conv3_bias:	ђC
'assignvariableop_20_block5_conv1_kernel:ђђ4
%assignvariableop_21_block5_conv1_bias:	ђC
'assignvariableop_22_block5_conv2_kernel:ђђ4
%assignvariableop_23_block5_conv2_bias:	ђC
'assignvariableop_24_block5_conv3_kernel:ђђ4
%assignvariableop_25_block5_conv3_bias:	ђ4
 assignvariableop_26_dense_kernel:
ђў-,
assignvariableop_27_dense_bias:-'
assignvariableop_28_iteration:	 +
!assignvariableop_29_learning_rate: ;
'assignvariableop_30_adam_m_dense_kernel:
ђў-;
'assignvariableop_31_adam_v_dense_kernel:
ђў-3
%assignvariableop_32_adam_m_dense_bias:-3
%assignvariableop_33_adam_v_dense_bias:-%
assignvariableop_34_total_1: %
assignvariableop_35_count_1: #
assignvariableop_36_total: #
assignvariableop_37_count: 
identity_39ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9ц
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*╩
value└Bй'B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЙ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*▓
_output_shapesЪ
ю:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block4_conv1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block4_conv1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv2_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv2_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv3_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv3_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block5_conv1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block5_conv1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block5_conv2_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block5_conv2_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv3_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv3_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_26AssignVariableOp assignvariableop_26_dense_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_27AssignVariableOpassignvariableop_27_dense_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:Х
AssignVariableOp_28AssignVariableOpassignvariableop_28_iterationIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_29AssignVariableOp!assignvariableop_29_learning_rateIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_m_dense_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_v_dense_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_m_dense_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_v_dense_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_34AssignVariableOpassignvariableop_34_total_1Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_35AssignVariableOpassignvariableop_35_count_1Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_36AssignVariableOpassignvariableop_36_totalIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_37AssignVariableOpassignvariableop_37_countIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Њ
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_39IdentityIdentity_38:output:0^NoOp_1*
T0*
_output_shapes
: ђ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_39Identity_39:output:0*a
_input_shapesP
N: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
К
]
A__inference_flatten_layer_call_and_return_conditional_losses_4126

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     ї  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђўZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђў"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         
ђ:X T
0
_output_shapes
:         
ђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2722

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Й
Њ
$__inference_dense_layer_call_fn_4135

inputs
unknown:
ђў-
	unknown_0:-
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2765o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђў: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         ђў
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block4_conv2_layer_call_and_return_conditional_losses_4015

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         (ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         (ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
╝
Ж
$__inference_model_layer_call_fn_3526

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ&

unknown_13:ђђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ&

unknown_19:ђђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ&

unknown_23:ђђ

unknown_24:	ђ

unknown_25:
ђў-

unknown_26:-
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2705

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2670

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         (ђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         (ђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
Г
B
&__inference_flatten_layer_call_fn_4120

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђў* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_2752b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ђў"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         
ђ:X T
0
_output_shapes
:         
ђ
 
_user_specified_nameinputs
Ї
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_3855

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш
Б
+__inference_block4_conv2_layer_call_fn_4004

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         (ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2670x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         (ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         (ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         (ђ
 
_user_specified_nameinputs
Љ
 
F__inference_block1_conv1_layer_call_and_return_conditional_losses_3825

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ­└@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ­└@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ­└: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
ш
Б
+__inference_block5_conv3_layer_call_fn_4094

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2739x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
╝
Ж
$__inference_model_layer_call_fn_3587

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ&

unknown_13:ђђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ&

unknown_19:ђђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ&

unknown_23:ђђ

unknown_24:	ђ

unknown_25:
ђў-

unknown_26:-
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_3120o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
Ї
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_2486

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┐
в
$__inference_model_layer_call_fn_2831
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ&

unknown_13:ђђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ&

unknown_19:ђђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ&

unknown_23:ђђ

unknown_24:	ђ

unknown_25:
ђў-

unknown_26:-
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ­└
!
_user_specified_name	input_1
Ш
а
+__inference_block1_conv1_layer_call_fn_3814

inputs!
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ­└@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2531y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ­└@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ­└: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
Ї
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_2474

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Љ
ѓ
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2618

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <PђY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         <Pђj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         <Pђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         <Pђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         <Pђ
 
_user_specified_nameinputs
Б

Ы
?__inference_dense_layer_call_and_return_conditional_losses_4146

inputs2
matmul_readvariableop_resource:
ђў--
biasadd_readvariableop_resource:-
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђў-*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         -`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         -w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђў: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         ђў
 
_user_specified_nameinputs
Ї
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_2498

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ю
ж
"__inference_signature_wrapper_3465
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ&

unknown_13:ђђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ&

unknown_19:ђђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ&

unknown_23:ђђ

unknown_24:	ђ

unknown_25:
ђў-

unknown_26:-
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         -*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__wrapped_model_2453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         -`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ­└
!
_user_specified_name	input_1
зљ
а
?__inference_model_layer_call_and_return_conditional_losses_3696

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@ђ;
,block2_conv1_biasadd_readvariableop_resource:	ђG
+block2_conv2_conv2d_readvariableop_resource:ђђ;
,block2_conv2_biasadd_readvariableop_resource:	ђG
+block3_conv1_conv2d_readvariableop_resource:ђђ;
,block3_conv1_biasadd_readvariableop_resource:	ђG
+block3_conv2_conv2d_readvariableop_resource:ђђ;
,block3_conv2_biasadd_readvariableop_resource:	ђG
+block3_conv3_conv2d_readvariableop_resource:ђђ;
,block3_conv3_biasadd_readvariableop_resource:	ђG
+block4_conv1_conv2d_readvariableop_resource:ђђ;
,block4_conv1_biasadd_readvariableop_resource:	ђG
+block4_conv2_conv2d_readvariableop_resource:ђђ;
,block4_conv2_biasadd_readvariableop_resource:	ђG
+block4_conv3_conv2d_readvariableop_resource:ђђ;
,block4_conv3_biasadd_readvariableop_resource:	ђG
+block5_conv1_conv2d_readvariableop_resource:ђђ;
,block5_conv1_biasadd_readvariableop_resource:	ђG
+block5_conv2_conv2d_readvariableop_resource:ђђ;
,block5_conv2_biasadd_readvariableop_resource:	ђG
+block5_conv3_conv2d_readvariableop_resource:ђђ;
,block5_conv3_biasadd_readvariableop_resource:	ђ8
$dense_matmul_readvariableop_resource:
ђў-3
%dense_biasadd_readvariableop_resource:-
identityѕб#block1_conv1/BiasAdd/ReadVariableOpб"block1_conv1/Conv2D/ReadVariableOpб#block1_conv2/BiasAdd/ReadVariableOpб"block1_conv2/Conv2D/ReadVariableOpб#block2_conv1/BiasAdd/ReadVariableOpб"block2_conv1/Conv2D/ReadVariableOpб#block2_conv2/BiasAdd/ReadVariableOpб"block2_conv2/Conv2D/ReadVariableOpб#block3_conv1/BiasAdd/ReadVariableOpб"block3_conv1/Conv2D/ReadVariableOpб#block3_conv2/BiasAdd/ReadVariableOpб"block3_conv2/Conv2D/ReadVariableOpб#block3_conv3/BiasAdd/ReadVariableOpб"block3_conv3/Conv2D/ReadVariableOpб#block4_conv1/BiasAdd/ReadVariableOpб"block4_conv1/Conv2D/ReadVariableOpб#block4_conv2/BiasAdd/ReadVariableOpб"block4_conv2/Conv2D/ReadVariableOpб#block4_conv3/BiasAdd/ReadVariableOpб"block4_conv3/Conv2D/ReadVariableOpб#block5_conv1/BiasAdd/ReadVariableOpб"block5_conv1/Conv2D/ReadVariableOpб#block5_conv2/BiasAdd/ReadVariableOpб"block5_conv2/Conv2D/ReadVariableOpб#block5_conv3/BiasAdd/ReadVariableOpб"block5_conv3/Conv2D/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpќ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0х
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
ї
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ­└@ќ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╬
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@*
paddingSAME*
strides
ї
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ­└@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         ­└@Г
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*0
_output_shapes
:         xа@*
ksize
*
paddingVALID*
strides
Ќ
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0╦
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
Ї
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0д
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађt
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         xађў
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╬
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
Ї
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0д
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађt
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         xађГ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:         <Pђ*
ksize
*
paddingVALID*
strides
ў
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╩
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ї
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         <Pђў
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ї
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         <Pђў
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђ*
paddingSAME*
strides
Ї
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         <Pђs
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         <PђГ
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:         (ђ*
ksize
*
paddingVALID*
strides
ў
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╩
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ї
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђў
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ї
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђў
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђ*
paddingSAME*
strides
Ї
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         (ђs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         (ђГ
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
ў
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╩
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ї
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         ђў
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ї
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         ђў
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0═
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ї
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ц
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         ђГ
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:         
ђ*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     ї  ё
flatten/ReshapeReshapeblock5_pool/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:         ђўѓ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђў-*
dtype0Є
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         -b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:         -f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         -м
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ­└: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ­└
 
_user_specified_nameinputs
Њ
Ђ
F__inference_block2_conv1_layer_call_and_return_conditional_losses_3875

inputs9
conv2d_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         xађZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         xађk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         xађw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         xа@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         xа@
 
_user_specified_nameinputs"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▓
serving_defaultъ
E
input_1:
serving_default_input_1:0         ­└9
dense0
StatefulPartitionedCall:0         -tensorflow/serving/predict:«Х
№
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer-19
layer_with_weights-13
layer-20
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
П
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
П
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op"
_tf_keras_layer
Ц
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
П
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op"
_tf_keras_layer
П
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias
 I_jit_compiled_convolution_op"
_tf_keras_layer
Ц
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
П
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias
 X_jit_compiled_convolution_op"
_tf_keras_layer
П
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

_kernel
`bias
 a_jit_compiled_convolution_op"
_tf_keras_layer
П
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op"
_tf_keras_layer
Ц
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
П
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias
 y_jit_compiled_convolution_op"
_tf_keras_layer
Я
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
ђkernel
	Ђbias
!ѓ_jit_compiled_convolution_op"
_tf_keras_layer
Т
Ѓ	variables
ёtrainable_variables
Ёregularization_losses
є	keras_api
Є__call__
+ѕ&call_and_return_all_conditional_losses
Ѕkernel
	іbias
!І_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
Т
њ	variables
Њtrainable_variables
ћregularization_losses
Ћ	keras_api
ќ__call__
+Ќ&call_and_return_all_conditional_losses
ўkernel
	Ўbias
!џ_jit_compiled_convolution_op"
_tf_keras_layer
Т
Џ	variables
юtrainable_variables
Юregularization_losses
ъ	keras_api
Ъ__call__
+а&call_and_return_all_conditional_losses
Аkernel
	бbias
!Б_jit_compiled_convolution_op"
_tf_keras_layer
Т
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
е__call__
+Е&call_and_return_all_conditional_losses
фkernel
	Фbias
!г_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Г	variables
«trainable_variables
»regularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
│	variables
┤trainable_variables
хregularization_losses
Х	keras_api
и__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
├
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
й__call__
+Й&call_and_return_all_conditional_losses
┐kernel
	└bias"
_tf_keras_layer
ѓ
&0
'1
/2
03
>4
?5
G6
H7
V8
W9
_10
`11
h12
i13
w14
x15
ђ16
Ђ17
Ѕ18
і19
ў20
Ў21
А22
б23
ф24
Ф25
┐26
└27"
trackable_list_wrapper
0
┐0
└1"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
═
кtrace_0
Кtrace_1
╚trace_2
╔trace_32┌
$__inference_model_layer_call_fn_2831
$__inference_model_layer_call_fn_3526
$__inference_model_layer_call_fn_3587
$__inference_model_layer_call_fn_3240┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zкtrace_0zКtrace_1z╚trace_2z╔trace_3
╣
╩trace_0
╦trace_1
╠trace_2
═trace_32к
?__inference_model_layer_call_and_return_conditional_losses_3696
?__inference_model_layer_call_and_return_conditional_losses_3805
?__inference_model_layer_call_and_return_conditional_losses_3320
?__inference_model_layer_call_and_return_conditional_losses_3400┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╩trace_0z╦trace_1z╠trace_2z═trace_3
╩BК
__inference__wrapped_model_2453input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Б
╬
_variables
¤_iterations
л_learning_rate
Л_index_dict
м
_momentums
М_velocities
н_update_step_xla"
experimentalOptimizer
-
Нserving_default"
signature_map
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
оnon_trainable_variables
Оlayers
пmetrics
 ┘layer_regularization_losses
┌layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ы
█trace_02м
+__inference_block1_conv1_layer_call_fn_3814б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z█trace_0
ї
▄trace_02ь
F__inference_block1_conv1_layer_call_and_return_conditional_losses_3825б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▄trace_0
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ы
Рtrace_02м
+__inference_block1_conv2_layer_call_fn_3834б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zРtrace_0
ї
сtrace_02ь
F__inference_block1_conv2_layer_call_and_return_conditional_losses_3845б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zсtrace_0
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Сnon_trainable_variables
тlayers
Тmetrics
 уlayer_regularization_losses
Уlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
­
жtrace_02Л
*__inference_block1_pool_layer_call_fn_3850б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zжtrace_0
І
Жtrace_02В
E__inference_block1_pool_layer_call_and_return_conditional_losses_3855б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЖtrace_0
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ы
­trace_02м
+__inference_block2_conv1_layer_call_fn_3864б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z­trace_0
ї
ыtrace_02ь
F__inference_block2_conv1_layer_call_and_return_conditional_losses_3875б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zыtrace_0
.:,@ђ2block2_conv1/kernel
 :ђ2block2_conv1/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ыnon_trainable_variables
зlayers
Зmetrics
 шlayer_regularization_losses
Шlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
ы
эtrace_02м
+__inference_block2_conv2_layer_call_fn_3884б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zэtrace_0
ї
Эtrace_02ь
F__inference_block2_conv2_layer_call_and_return_conditional_losses_3895б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЭtrace_0
/:-ђђ2block2_conv2/kernel
 :ђ2block2_conv2/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
щnon_trainable_variables
Щlayers
чmetrics
 Чlayer_regularization_losses
§layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
­
■trace_02Л
*__inference_block2_pool_layer_call_fn_3900б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z■trace_0
І
 trace_02В
E__inference_block2_pool_layer_call_and_return_conditional_losses_3905б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z trace_0
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ђnon_trainable_variables
Ђlayers
ѓmetrics
 Ѓlayer_regularization_losses
ёlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ы
Ёtrace_02м
+__inference_block3_conv1_layer_call_fn_3914б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁtrace_0
ї
єtrace_02ь
F__inference_block3_conv1_layer_call_and_return_conditional_losses_3925б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zєtrace_0
/:-ђђ2block3_conv1/kernel
 :ђ2block3_conv1/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Єnon_trainable_variables
ѕlayers
Ѕmetrics
 іlayer_regularization_losses
Іlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
ы
їtrace_02м
+__inference_block3_conv2_layer_call_fn_3934б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zїtrace_0
ї
Їtrace_02ь
F__inference_block3_conv2_layer_call_and_return_conditional_losses_3945б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇtrace_0
/:-ђђ2block3_conv2/kernel
 :ђ2block3_conv2/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
ы
Њtrace_02м
+__inference_block3_conv3_layer_call_fn_3954б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊtrace_0
ї
ћtrace_02ь
F__inference_block3_conv3_layer_call_and_return_conditional_losses_3965б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zћtrace_0
/:-ђђ2block3_conv3/kernel
 :ђ2block3_conv3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ћnon_trainable_variables
ќlayers
Ќmetrics
 ўlayer_regularization_losses
Ўlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
­
џtrace_02Л
*__inference_block3_pool_layer_call_fn_3970б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zџtrace_0
І
Џtrace_02В
E__inference_block3_pool_layer_call_and_return_conditional_losses_3975б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЏtrace_0
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
юnon_trainable_variables
Юlayers
ъmetrics
 Ъlayer_regularization_losses
аlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
ы
Аtrace_02м
+__inference_block4_conv1_layer_call_fn_3984б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zАtrace_0
ї
бtrace_02ь
F__inference_block4_conv1_layer_call_and_return_conditional_losses_3995б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zбtrace_0
/:-ђђ2block4_conv1/kernel
 :ђ2block4_conv1/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
ђ0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ы
еtrace_02м
+__inference_block4_conv2_layer_call_fn_4004б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zеtrace_0
ї
Еtrace_02ь
F__inference_block4_conv2_layer_call_and_return_conditional_losses_4015б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЕtrace_0
/:-ђђ2block4_conv2/kernel
 :ђ2block4_conv2/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Ѕ0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
фnon_trainable_variables
Фlayers
гmetrics
 Гlayer_regularization_losses
«layer_metrics
Ѓ	variables
ёtrainable_variables
Ёregularization_losses
Є__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
ы
»trace_02м
+__inference_block4_conv3_layer_call_fn_4024б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z»trace_0
ї
░trace_02ь
F__inference_block4_conv3_layer_call_and_return_conditional_losses_4035б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z░trace_0
/:-ђђ2block4_conv3/kernel
 :ђ2block4_conv3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
хlayer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
­
Хtrace_02Л
*__inference_block4_pool_layer_call_fn_4040б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zХtrace_0
І
иtrace_02В
E__inference_block4_pool_layer_call_and_return_conditional_losses_4045б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zиtrace_0
0
ў0
Ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Иnon_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
њ	variables
Њtrainable_variables
ћregularization_losses
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
ы
йtrace_02м
+__inference_block5_conv1_layer_call_fn_4054б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zйtrace_0
ї
Йtrace_02ь
F__inference_block5_conv1_layer_call_and_return_conditional_losses_4065б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЙtrace_0
/:-ђђ2block5_conv1/kernel
 :ђ2block5_conv1/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
А0
б1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
Џ	variables
юtrainable_variables
Юregularization_losses
Ъ__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
ы
─trace_02м
+__inference_block5_conv2_layer_call_fn_4074б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z─trace_0
ї
┼trace_02ь
F__inference_block5_conv2_layer_call_and_return_conditional_losses_4085б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┼trace_0
/:-ђђ2block5_conv2/kernel
 :ђ2block5_conv2/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
ф0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
кnon_trainable_variables
Кlayers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
ц	variables
Цtrainable_variables
дregularization_losses
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
ы
╦trace_02м
+__inference_block5_conv3_layer_call_fn_4094б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╦trace_0
ї
╠trace_02ь
F__inference_block5_conv3_layer_call_and_return_conditional_losses_4105б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╠trace_0
/:-ђђ2block5_conv3/kernel
 :ђ2block5_conv3/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
═non_trainable_variables
╬layers
¤metrics
 лlayer_regularization_losses
Лlayer_metrics
Г	variables
«trainable_variables
»regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
­
мtrace_02Л
*__inference_block5_pool_layer_call_fn_4110б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zмtrace_0
І
Мtrace_02В
E__inference_block5_pool_layer_call_and_return_conditional_losses_4115б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zМtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
нnon_trainable_variables
Нlayers
оmetrics
 Оlayer_regularization_losses
пlayer_metrics
│	variables
┤trainable_variables
хregularization_losses
и__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
В
┘trace_02═
&__inference_flatten_layer_call_fn_4120б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┘trace_0
Є
┌trace_02У
A__inference_flatten_layer_call_and_return_conditional_losses_4126б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┌trace_0
0
┐0
└1"
trackable_list_wrapper
0
┐0
└1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
█non_trainable_variables
▄layers
Пmetrics
 яlayer_regularization_losses
▀layer_metrics
╣	variables
║trainable_variables
╗regularization_losses
й__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
Ж
Яtrace_02╦
$__inference_dense_layer_call_fn_4135б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЯtrace_0
Ё
рtrace_02Т
?__inference_dense_layer_call_and_return_conditional_losses_4146б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zрtrace_0
 :
ђў-2dense/kernel
:-2
dense/bias
­
&0
'1
/2
03
>4
?5
G6
H7
V8
W9
_10
`11
h12
i13
w14
x15
ђ16
Ђ17
Ѕ18
і19
ў20
Ў21
А22
б23
ф24
Ф25"
trackable_list_wrapper
Й
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
0
Р0
с1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ШBз
$__inference_model_layer_call_fn_2831input_1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
шBЫ
$__inference_model_layer_call_fn_3526inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
шBЫ
$__inference_model_layer_call_fn_3587inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
$__inference_model_layer_call_fn_3240input_1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
?__inference_model_layer_call_and_return_conditional_losses_3696inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
?__inference_model_layer_call_and_return_conditional_losses_3805inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
?__inference_model_layer_call_and_return_conditional_losses_3320input_1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
?__inference_model_layer_call_and_return_conditional_losses_3400input_1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
H
¤0
С1
т2
Т3
у4"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
0
С0
Т1"
trackable_list_wrapper
0
т0
у1"
trackable_list_wrapper
┐2╝╣
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
╔Bк
"__inference_signature_wrapper_3465input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block1_conv1_layer_call_fn_3814inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block1_conv1_layer_call_and_return_conditional_losses_3825inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block1_conv2_layer_call_fn_3834inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block1_conv2_layer_call_and_return_conditional_losses_3845inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB█
*__inference_block1_pool_layer_call_fn_3850inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_block1_pool_layer_call_and_return_conditional_losses_3855inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block2_conv1_layer_call_fn_3864inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block2_conv1_layer_call_and_return_conditional_losses_3875inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block2_conv2_layer_call_fn_3884inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block2_conv2_layer_call_and_return_conditional_losses_3895inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB█
*__inference_block2_pool_layer_call_fn_3900inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_block2_pool_layer_call_and_return_conditional_losses_3905inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block3_conv1_layer_call_fn_3914inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block3_conv1_layer_call_and_return_conditional_losses_3925inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block3_conv2_layer_call_fn_3934inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block3_conv2_layer_call_and_return_conditional_losses_3945inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block3_conv3_layer_call_fn_3954inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block3_conv3_layer_call_and_return_conditional_losses_3965inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB█
*__inference_block3_pool_layer_call_fn_3970inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_block3_pool_layer_call_and_return_conditional_losses_3975inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block4_conv1_layer_call_fn_3984inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block4_conv1_layer_call_and_return_conditional_losses_3995inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
ђ0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block4_conv2_layer_call_fn_4004inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block4_conv2_layer_call_and_return_conditional_losses_4015inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
Ѕ0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block4_conv3_layer_call_fn_4024inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block4_conv3_layer_call_and_return_conditional_losses_4035inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB█
*__inference_block4_pool_layer_call_fn_4040inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_block4_pool_layer_call_and_return_conditional_losses_4045inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
ў0
Ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block5_conv1_layer_call_fn_4054inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block5_conv1_layer_call_and_return_conditional_losses_4065inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
А0
б1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block5_conv2_layer_call_fn_4074inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block5_conv2_layer_call_and_return_conditional_losses_4085inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
ф0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block5_conv3_layer_call_fn_4094inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_block5_conv3_layer_call_and_return_conditional_losses_4105inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB█
*__inference_block5_pool_layer_call_fn_4110inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_block5_pool_layer_call_and_return_conditional_losses_4115inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌BО
&__inference_flatten_layer_call_fn_4120inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
шBЫ
A__inference_flatten_layer_call_and_return_conditional_losses_4126inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBН
$__inference_dense_layer_call_fn_4135inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
зB­
?__inference_dense_layer_call_and_return_conditional_losses_4146inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
У	variables
ж	keras_api

Жtotal

вcount"
_tf_keras_metric
c
В	variables
ь	keras_api

Ьtotal

№count
­
_fn_kwargs"
_tf_keras_metric
%:#
ђў-2Adam/m/dense/kernel
%:#
ђў-2Adam/v/dense/kernel
:-2Adam/m/dense/bias
:-2Adam/v/dense/bias
0
Ж0
в1"
trackable_list_wrapper
.
У	variables"
_generic_user_object
:  (2total
:  (2count
0
Ь0
№1"
trackable_list_wrapper
.
В	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper╣
__inference__wrapped_model_2453Ћ(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└:б7
0б-
+і(
input_1         ­└
ф "-ф*
(
denseі
dense         -┴
F__inference_block1_conv1_layer_call_and_return_conditional_losses_3825w&'9б6
/б,
*і'
inputs         ­└
ф "6б3
,і)
tensor_0         ­└@
џ Џ
+__inference_block1_conv1_layer_call_fn_3814l&'9б6
/б,
*і'
inputs         ­└
ф "+і(
unknown         ­└@┴
F__inference_block1_conv2_layer_call_and_return_conditional_losses_3845w/09б6
/б,
*і'
inputs         ­└@
ф "6б3
,і)
tensor_0         ­└@
џ Џ
+__inference_block1_conv2_layer_call_fn_3834l/09б6
/б,
*і'
inputs         ­└@
ф "+і(
unknown         ­└@№
E__inference_block1_pool_layer_call_and_return_conditional_losses_3855ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ ╔
*__inference_block1_pool_layer_call_fn_3850џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    └
F__inference_block2_conv1_layer_call_and_return_conditional_losses_3875v>?8б5
.б+
)і&
inputs         xа@
ф "6б3
,і)
tensor_0         xађ
џ џ
+__inference_block2_conv1_layer_call_fn_3864k>?8б5
.б+
)і&
inputs         xа@
ф "+і(
unknown         xађ┴
F__inference_block2_conv2_layer_call_and_return_conditional_losses_3895wGH9б6
/б,
*і'
inputs         xађ
ф "6б3
,і)
tensor_0         xађ
џ Џ
+__inference_block2_conv2_layer_call_fn_3884lGH9б6
/б,
*і'
inputs         xађ
ф "+і(
unknown         xађ№
E__inference_block2_pool_layer_call_and_return_conditional_losses_3905ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ ╔
*__inference_block2_pool_layer_call_fn_3900џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    ┐
F__inference_block3_conv1_layer_call_and_return_conditional_losses_3925uVW8б5
.б+
)і&
inputs         <Pђ
ф "5б2
+і(
tensor_0         <Pђ
џ Ў
+__inference_block3_conv1_layer_call_fn_3914jVW8б5
.б+
)і&
inputs         <Pђ
ф "*і'
unknown         <Pђ┐
F__inference_block3_conv2_layer_call_and_return_conditional_losses_3945u_`8б5
.б+
)і&
inputs         <Pђ
ф "5б2
+і(
tensor_0         <Pђ
џ Ў
+__inference_block3_conv2_layer_call_fn_3934j_`8б5
.б+
)і&
inputs         <Pђ
ф "*і'
unknown         <Pђ┐
F__inference_block3_conv3_layer_call_and_return_conditional_losses_3965uhi8б5
.б+
)і&
inputs         <Pђ
ф "5б2
+і(
tensor_0         <Pђ
џ Ў
+__inference_block3_conv3_layer_call_fn_3954jhi8б5
.б+
)і&
inputs         <Pђ
ф "*і'
unknown         <Pђ№
E__inference_block3_pool_layer_call_and_return_conditional_losses_3975ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ ╔
*__inference_block3_pool_layer_call_fn_3970џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    ┐
F__inference_block4_conv1_layer_call_and_return_conditional_losses_3995uwx8б5
.б+
)і&
inputs         (ђ
ф "5б2
+і(
tensor_0         (ђ
џ Ў
+__inference_block4_conv1_layer_call_fn_3984jwx8б5
.б+
)і&
inputs         (ђ
ф "*і'
unknown         (ђ┴
F__inference_block4_conv2_layer_call_and_return_conditional_losses_4015wђЂ8б5
.б+
)і&
inputs         (ђ
ф "5б2
+і(
tensor_0         (ђ
џ Џ
+__inference_block4_conv2_layer_call_fn_4004lђЂ8б5
.б+
)і&
inputs         (ђ
ф "*і'
unknown         (ђ┴
F__inference_block4_conv3_layer_call_and_return_conditional_losses_4035wЅі8б5
.б+
)і&
inputs         (ђ
ф "5б2
+і(
tensor_0         (ђ
џ Џ
+__inference_block4_conv3_layer_call_fn_4024lЅі8б5
.б+
)і&
inputs         (ђ
ф "*і'
unknown         (ђ№
E__inference_block4_pool_layer_call_and_return_conditional_losses_4045ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ ╔
*__inference_block4_pool_layer_call_fn_4040џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    ┴
F__inference_block5_conv1_layer_call_and_return_conditional_losses_4065wўЎ8б5
.б+
)і&
inputs         ђ
ф "5б2
+і(
tensor_0         ђ
џ Џ
+__inference_block5_conv1_layer_call_fn_4054lўЎ8б5
.б+
)і&
inputs         ђ
ф "*і'
unknown         ђ┴
F__inference_block5_conv2_layer_call_and_return_conditional_losses_4085wАб8б5
.б+
)і&
inputs         ђ
ф "5б2
+і(
tensor_0         ђ
џ Џ
+__inference_block5_conv2_layer_call_fn_4074lАб8б5
.б+
)і&
inputs         ђ
ф "*і'
unknown         ђ┴
F__inference_block5_conv3_layer_call_and_return_conditional_losses_4105wфФ8б5
.б+
)і&
inputs         ђ
ф "5б2
+і(
tensor_0         ђ
џ Џ
+__inference_block5_conv3_layer_call_fn_4094lфФ8б5
.б+
)і&
inputs         ђ
ф "*і'
unknown         ђ№
E__inference_block5_pool_layer_call_and_return_conditional_losses_4115ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ ╔
*__inference_block5_pool_layer_call_fn_4110џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    ф
?__inference_dense_layer_call_and_return_conditional_losses_4146g┐└1б.
'б$
"і
inputs         ђў
ф ",б)
"і
tensor_0         -
џ ё
$__inference_dense_layer_call_fn_4135\┐└1б.
'б$
"і
inputs         ђў
ф "!і
unknown         -»
A__inference_flatten_layer_call_and_return_conditional_losses_4126j8б5
.б+
)і&
inputs         
ђ
ф ".б+
$і!
tensor_0         ђў
џ Ѕ
&__inference_flatten_layer_call_fn_4120_8б5
.б+
)і&
inputs         
ђ
ф "#і 
unknown         ђўЯ
?__inference_model_layer_call_and_return_conditional_losses_3320ю(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└Bб?
8б5
+і(
input_1         ­└
p 

 
ф ",б)
"і
tensor_0         -
џ Я
?__inference_model_layer_call_and_return_conditional_losses_3400ю(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└Bб?
8б5
+і(
input_1         ­└
p

 
ф ",б)
"і
tensor_0         -
џ ▀
?__inference_model_layer_call_and_return_conditional_losses_3696Џ(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└Aб>
7б4
*і'
inputs         ­└
p 

 
ф ",б)
"і
tensor_0         -
џ ▀
?__inference_model_layer_call_and_return_conditional_losses_3805Џ(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└Aб>
7б4
*і'
inputs         ­└
p

 
ф ",б)
"і
tensor_0         -
џ ║
$__inference_model_layer_call_fn_2831Љ(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└Bб?
8б5
+і(
input_1         ­└
p 

 
ф "!і
unknown         -║
$__inference_model_layer_call_fn_3240Љ(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└Bб?
8б5
+і(
input_1         ­└
p

 
ф "!і
unknown         -╣
$__inference_model_layer_call_fn_3526љ(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└Aб>
7б4
*і'
inputs         ­└
p 

 
ф "!і
unknown         -╣
$__inference_model_layer_call_fn_3587љ(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└Aб>
7б4
*і'
inputs         ­└
p

 
ф "!і
unknown         -К
"__inference_signature_wrapper_3465а(&'/0>?GHVW_`hiwxђЂЅіўЎАбфФ┐└EбB
б 
;ф8
6
input_1+і(
input_1         ­└"-ф*
(
denseі
dense         -