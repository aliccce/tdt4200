#include "bitmap.h"

/* save 24-bits bmp file, buffer must be in bmp format: upside-down */
void savebmp(char *name,uchar *buffer,int x,int y) {
	FILE *f=fopen(name,"wb");
	if(!f) {
		printf("Error writing image to disk.\n");
		return;
	}
	unsigned int size=x*y*3+54;
	uchar header[54]={'B','M',size&255,(size>>8)&255,(size>>16)&255,size>>24,0,
                    0,0,0,54,0,0,0,40,0,0,0,x&255,x>>8,0,0,y&255,y>>8,0,0,1,0,24,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	fwrite(header,1,54,f);
	fwrite(buffer,1,XSIZE*YSIZE*3,f);
	fclose(f);
}


/* given iteration number, set a colour */
void fancycolour(uchar *p,int iter) {
	if(iter==MAXITER);
	else if(iter<8) { p[0]=128+iter*16; p[1]=p[2]=0; }
	else if(iter<24) { p[0]=255; p[1]=p[2]=(iter-8)*16; }
	else if(iter<160) { p[0]=p[1]=255-(iter-24)*2; p[2]=255; }
	else { p[0]=p[1]=(iter-160)*2; p[2]=255-(iter-160)*2; }
}
