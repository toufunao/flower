#!/bin/bash
sh_list=(

)
USER=
PASSWORD=
# ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa &> /dev/null
for(( i=0;i<${#sh_list[@]};i++))
do
{
echo "copy to ${sh_list[i]} : "
expect <<EOF
spawn ssh-copy-id -i ${USER}@${sh_list[i]}
expect {
  "yes/no" { send "yes\n";exp_continue }
  "password" { send "${PASSWORD}\n" }
}
expect eof
EOF
}&
done
wait
