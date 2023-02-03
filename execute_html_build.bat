call D:/anaconda3/Scripts/activate.bat D:/anaconda3/
call conda activate general_ds
call cd /d D:\LEARN\DATA SCIENCE\2 WORKSPACE\4 PROJECT\0 BOOK\mybookcollection\creditdefaultml
call rmdir /s /q "_build"
call cd /d D:\LEARN\DATA SCIENCE\2 WORKSPACE\4 PROJECT\1 LOAN RISK PREDICTION\creditdefaultml
call cd ..
call jupyter-book build --path-output "D:\LEARN\DATA SCIENCE\2 WORKSPACE\4 PROJECT\0 BOOK\mybookcollection\creditdefaultml" creditdefaultml
