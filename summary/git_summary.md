# git rm --cached
删除远程文件，工作区不删除
用-r参数删除目录, git rm --cached a.txt 删除的是本地仓库中的文件，且本地工作区的文件会保留且不再与远程仓库发生跟踪关系，如果本地仓库中的文件也要删除则用git rm a.txt

# git branch
查看分支

# git checkout -b whitespace
Switched to a new branch 'whitespace'

# git status 
查看状态

# git branch -d <branch_name>
如果需要删除的分支不是当前正在打开的分支，使用branch -d直接删除

# git branch -D <branch_name>
删除一个正打开的分支