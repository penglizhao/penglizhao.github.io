---
Layout: default
title: Useful CodeLine
---
## Matlab
判断标志
```
isnan();   % Judge for NAN
isempty(); % Judge for []
```
矩阵中寻找
```
find(x<5,1,'last'); % find last less than 5 in matrix X
find(x==5)
```
按照条件处理矩阵
```
 A(A<9 & A>2)
```

画图相关
```
plotyy(X1,Y1,X2,Y2,function);

text(x, y, 'text', 'horizontalAlignment', 'halign', ...
  'verticalAlignment', 'valign');
```
---
[Click to go back](https://zhaoph2008.github.io/)