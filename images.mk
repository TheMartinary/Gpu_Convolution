##############################
#
# Setup test images and check target
#

#URLS of images to test
IMAGE_URLS= 	http://farm1.static.flickr.com/222/447846619_ea0e3c9609.jpg \
	        	http://farm3.static.flickr.com/2379/2087393988_be21769f95.jpg \
				http://farm1.static.flickr.com/30/59838354_052062bf8c.jpg \
				http://farm1.static.flickr.com/201/456440842_a39b783d7b.jpg \
				http://farm1.static.flickr.com/180/421280328_317f6103ec.jpg \
				http://farm3.static.flickr.com/2257/2165764491_b7efd35ab1.jpg \
				http://farm1.static.flickr.com/177/367989583_10c1f2fc38.jpg \
				http://farm1.static.flickr.com/52/139935656_50bc16c15b.jpg

#Classification index of test images (in order)
CLASS_IDX=  	281\
				281\
				285\
				291\
				728\
				279\
				285\
				281

# Getters
JOINED   = $(join $(addsuffix @,$(IMAGE_URLS)),$(CLASS_IDX))
GET_URL  = $(word 1,$(subst @, ,$1))
GET_IDX  = $(word 2,$(subst @, ,$1))

#Bash coloring
RED=\033[0;31m
GREEN=\033[0;32m
NC=\033[0m

#$1=URL $2=NAME $3=CONVERTED_PNG $4=CHECK $5=IDX
define IMAGE_BUILD_RULES

#download image
$2:
	wget "$(strip $1)" -O $2

#convert
$3:$2
	convert $2 -resize 224x224! $3

#check if correct class is identified. If not error
$4:$3 $(EXE)
	@echo "Evaluating image $3"
	./$(EXE) $3 | tee $4
	@grep -q "Detected class: $(strip $5)" $4 && printf "$(GREEN)correctly identified image $2$(NC)\n" ||  (printf "$(RED)Did not correctly identify image $2$(NC)\n"; rm -f $4; exit 1)

endef

#check if all images are classified correctly
check_all: $(foreach URL, $(IMAGE_URLS), check_$(basename $(notdir $(URL))))
	@printf "$(GREEN)All correct!$(NC)\n"

#define build rules for all images
$(foreach j,$(JOINED),$(eval $(call IMAGE_BUILD_RULES,\
	$(call GET_URL, $j),\
	$(notdir $(call GET_URL, $j)),\
	converted_$(basename $(notdir $(call GET_URL, $j))).png,\
	check_$(basename $(notdir $(call GET_URL, $j))),\
	$(call GET_IDX,$j)\
)))
