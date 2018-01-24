#!/bin/bash

SOURCE=module-xml
DESTINATION=generated-from-xml

# clean-up
rm -rf doc
rm -rf $DESTINATION

# generate doxy from xml
mkdir $DESTINATION
list=`find $SOURCE -iname *.xml | xargs`
for i in $list
do
   filename=`basename $i`
   doxyfile=${filename%%.*}
   xsltproc --output $DESTINATION/$doxyfile.dox ~/GitHub/yarp/scripts/yarp-module.xsl $i
done

doxygen ./generate.txt
