# Artetxe Embs are available in: En-> It, De, Fi, Es
for lang in it de fi es 
do
       for size in 5000 10000 20000 50000
       do	       
	       sh run.sh en $lang $size
       done
done

