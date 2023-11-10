OPENQASM 2.0;
include "qelib1.inc";

qreg a[3];
creg c[3];

x a[0];
x a[1];
h a[2];
cx a[1],a[2];
tdg a[2];
cx a[0],a[2];
t a[2];
cx a[1],a[2];
tdg a[2];
cx a[0],a[2];
tdg a[1];
t a[2];
cx a[0],a[1];
h a[2];
tdg a[1];
cx a[0],a[1];
t a[0];
s a[1];
measure a[0] -> c[0];
measure a[1] -> c[1];
measure a[2] -> c[2];
