#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>       
#define PI 3.14159265

//base calculation to operate convolution
void convolution_base(int *redDest, int *greenDest, int *blueDest,
		const int *redSource, const int *greenSource, const int *blueSource,
		const int n, const int m, float k[25]){
	int i, j, index;	
	#pragma omp parallel for schedule(static, n)
	#pragma omp simd
	for (i = 2; i < n-2; i++)
    	for (j = 2; j < m-2; j++){
            index = j + i * m;
           	    redDest[index]   =  k[0] *redSource[index-2*m-2]+k[1] *redSource[index-2*m-1]+k[2] *redSource[index-2*m]+k[3] *redSource[index-2*m+1]+k[4] *redSource[index-2*m+2]+
									k[5] *redSource[index-m-2]  +k[6] *redSource[index-m-1]  +k[7] *redSource[index-m]  +k[8] *redSource[index-m+1]  +k[9] *redSource[index-m+2]  +
									k[10]*redSource[index-2]    +k[11]*redSource[index-1]    +k[12]*redSource[index]    +k[13]*redSource[index+1]    +k[14]*redSource[index+2]    +
									k[15]*redSource[index+m-2]  +k[16]*redSource[index+m-1]  +k[17]*redSource[index+m]  +k[18]*redSource[index+m+1]  +k[19]*redSource[index+m+2]  +
									k[20]*redSource[index+2*m-2]+k[21]*redSource[index+2*m-1]+k[22]*redSource[index+2*m]+k[23]*redSource[index+2*m+1]+k[24]*redSource[index+2*m+2];
							    
			    greenDest[index] =  k[0] *greenSource[index-2*m-2]+k[1] *greenSource[index-2*m-1]+k[2] *greenSource[index-2*m]+k[3] *greenSource[index-2*m+1]+k[4] *greenSource[index-2*m+2]+
									k[5] *greenSource[index-m-2]  +k[6] *greenSource[index-m-1]  +k[7] *greenSource[index-m]  +k[8] *greenSource[index-m+1]  +k[9] *greenSource[index-m+2]  +
									k[10]*greenSource[index-2]    +k[11]*greenSource[index-1]    +k[12]*greenSource[index]    +k[13]*greenSource[index+1]    +k[14]*greenSource[index+2]    +
									k[15]*greenSource[index+m-2]  +k[16]*greenSource[index+m-1]  +k[17]*greenSource[index+m]  +k[18]*greenSource[index+m+1]  +k[19]*greenSource[index+m+2]  +
									k[20]*greenSource[index+2*m-2]+k[21]*greenSource[index+2*m-1]+k[22]*greenSource[index+2*m]+k[23]*greenSource[index+2*m+1]+k[24]*greenSource[index+2*m+2];
								
			    blueDest[index]  =  k[0] *blueSource[index-2*m-2]+k[1] *blueSource[index-2*m-1]+k[2] *blueSource[index-2*m]+k[3] *blueSource[index-2*m+1]+k[4] *blueSource[index-2*m+2]+
									k[5] *blueSource[index-m-2]  +k[6] *blueSource[index-m-1]  +k[7] *blueSource[index-m]  +k[8] *blueSource[index-m+1]  +k[9] *blueSource[index-m+2]  +
									k[10]*blueSource[index-2]    +k[11]*blueSource[index-1]    +k[12]*blueSource[index]    +k[13]*blueSource[index+1]    +k[14]*blueSource[index+2]    +
									k[15]*blueSource[index+m-2]  +k[16]*blueSource[index+m-1]  +k[17]*blueSource[index+m]  +k[18]*blueSource[index+m+1]  +k[19]*blueSource[index+m+2]  +
									k[20]*blueSource[index+2*m-2]+k[21]*blueSource[index+2*m-1]+k[22]*blueSource[index+2*m]+k[23]*blueSource[index+2*m+1]+k[24]*blueSource[index+2*m+2];
}}

//same in black and white
void convolution_BW(int *greyDest, const int *greySource, const int n, const int m, float k[25]){
	int i, j, index;
	#pragma omp parallel for schedule(static, n)
	#pragma omp simd
	for (i = 2; i < n-2; i++)
    	for (j = 2; j < m-2; j++){
            index = j + i * m;
           	    greyDest[index]   =  k[0] *greySource[index-2*m-2]+k[1] *greySource[index-2*m-1]+k[2] *greySource[index-2*m]+k[3] *greySource[index-2*m+1]+k[4] *greySource[index-2*m+2]+
									 k[5] *greySource[index-m-2]  +k[6] *greySource[index-m-1]  +k[7] *greySource[index-m]  +k[8] *greySource[index-m+1]  +k[9] *greySource[index-m+2]  +
									 k[10]*greySource[index-2]    +k[11]*greySource[index-1]    +k[12]*greySource[index]    +k[13]*greySource[index+1]    +k[14]*greySource[index+2]    +
									 k[15]*greySource[index+m-2]  +k[16]*greySource[index+m-1]  +k[17]*greySource[index+m]  +k[18]*greySource[index+m+1]  +k[19]*greySource[index+m+2]  +
									 k[20]*greySource[index+2*m-2]+k[21]*greySource[index+2*m-1]+k[22]*greySource[index+2*m]+k[23]*greySource[index+2*m+1]+k[24]*greySource[index+2*m+2];
						}
	}


//function to prevent going over 255 value for any color on any pixel, which could cause file corruption
void securite(int *redDest, int *greenDest, int *blueDest, const int n, const int m)
	{
		int i, j, index;
		for (i = 0; i < n; i++)
			for (j = 0; j < m; j++)
				{
					index = j + i * m;
					if (redDest[index]<0)
						redDest[index]=0;
					if (greenDest[index]<0)
						greenDest[index]=0;
					if (blueDest[index]<0)
						blueDest[index]=0;
					if (redDest[index]>255)
						redDest[index]=255;
					if (greenDest[index]>255)
						greenDest[index]=255;
					if (blueDest[index]>255)
						blueDest[index]=255;
				}
				
	}


//same as previous for canny edge detection (only black and white)
void securite_BW(int *greyDest, const int n, const int m)
	{
		int i, j, index;
		for (i = 0; i < n; i++)
			for (j = 0; j < m; j++)
				{
					index = j + i * m;
					if (greyDest[index]<0)
						greyDest[index]=0;
					if (greyDest[index]>255)
						greyDest[index]=255;
				}
				
	}
	
//Kernels for image processing
float blur[25]={0,   0,     0,     0,   0,
			    0, 1./9., 1./9., 1./9., 0,
	            0, 1./9., 1./9., 1./9., 0,
	            0, 1./9., 1./9., 1./9., 0,
	            0,   0,     0,     0,   0};
	           
float copy[25]={0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
			    0, 0, 1, 0, 0,
			    0, 0, 0, 0, 0,
			    0, 0, 0, 0, 0};	    
	           
float edge[25]={0,  0,  0,  0, 0,
				0,  0, -1,  0, 0,
			    0, -1,  4, -1, 0,
			    0,  0, -1,  0, 0,
			    0,  0,  0,  0, 0};
			  
float sharp[25]={0,  0,  0,  0, 0,
				 0,  0, -1,  0, 0,
			     0, -1,  5, -1, 0,	
			     0,  0, -1,  0, 0,
			     0,  0,  0,  0, 0};		
				
float noise[25]={0,  0,  0,  0, 0,
				   0, -2,  0, -2, 0, 
				   0,  7, -5,  7, 0,
				   0, -2,  0, -2, 0,
				   0,  0,  0,  0, 0};
				 
float gauss[25]={2./115.,  4./115.,  5./115.,  4./115., 2./115.,
				 4./115.,  9./115., 12./115.,  9./115., 4./115.,
				 5./115., 12./115., 15./115., 12./115., 5./115.,
				 4./115.,  9./115., 12./115.,  9./115., 4./115.,
				 2./115.,  4./115.,  5./115.,  4./115., 2./115.};
			
//gradient x axis 	 
float gx[25]={0,  0, 0, 0, 0,
			  0, -1, 0, 1, 0,
			  0, -2, 0, 2, 0,
			  0, -1, 0, 1, 0,
			  0,  0, 0, 0, 0};

//gradient y axis
float gy[25]={0,  0,  0,  0, 0,
			  0,  1,  2,  1, 0,
			  0,  0,  0,  0, 0,
			  0, -1, -2, -1, 0,
			  0,  0,  0,  0, 0};

//thickens lines (usefull for large picture)
float thick[25]={0,   0,     0,     0,   0,
			     0, 3./9., 3./9., 3./9., 0,
	             0, 3./9., 3./9., 3./9., 0,
	             0, 3./9., 3./9., 3./9., 0,
	             0,   0,     0,     0,   0};	

int main(int argc, char *argv[] )
{	  
	//declarations
	int n, m, colorDepth, i, j, index;
	double time_fetch, time_convolution, time_writing, time_thinking;
	double stop1, stop2, stop3, start1, start2, start3, stop4, start4;
	int *redSource, *greenSource, *blueSource;
	int *redDest, *greenDest, *blueDest, *greyDest, *greyDest2;
	int *gradxR,*gradxG,*gradxB, *gradx;
	int *gradyR,*gradyG,*gradyB, *grady;
	int *edge_strenght, *nonMAX; 
	double edge_avg=0;
	int selector=42, thicker=0;
	int trH=1000, trL=1000;
	double *angle;
	FILE *inFile, *outFile;
	if (argc < 2 ){
		printf("Usage: convolution input_image output_image\n");
		exit(1);
	}
//----------------------------------------------------------------------
	start1 = omp_get_wtime();

	// Opening input file
	inFile = fopen(argv[1], "r");
	if (inFile == NULL) {
	  printf("Can't open input file %s\n", argv[1]);
	  exit(1);
	}

	// Reading magic number
	char magic_number[5];
    fscanf(inFile, "%s", magic_number);
    if (strcmp(magic_number, "P3")){
  	  printf("Error while reading file %s\n", argv[1]);
  	  exit(1);
    }
    // Reading image size
    fscanf(inFile, "%d", &m);
    fscanf(inFile, "%d", &n);
    // Reading color depth
    fscanf(inFile, "%d", &colorDepth);

    // Allocating memory
    redSource = (int*)malloc(n * m * sizeof(int));
    greenSource = (int*)malloc(n * m * sizeof(int));
    blueSource = (int*)malloc(n * m * sizeof(int));
    redDest = (int*)malloc(n * m * sizeof(int));
    greenDest = (int*)malloc(n * m * sizeof(int));
    blueDest = (int*)malloc(n * m * sizeof(int));
    greyDest=(int*)malloc(n*m*sizeof(int));
    greyDest2=(int*)malloc(n*m*sizeof(int));
    gradxR=(int*)malloc(n*m*sizeof(int));
    gradxG=(int*)malloc(n*m*sizeof(int));
    gradxB=(int*)malloc(n*m*sizeof(int));
    gradyR=(int*)malloc(n*m*sizeof(int));
    gradyG=(int*)malloc(n*m*sizeof(int));
    gradyB=(int*)malloc(n*m*sizeof(int));
    gradx=(int*)malloc(n*m*sizeof(int));
    grady=(int*)malloc(n*m*sizeof(int));
    edge_strenght=(int*)malloc(n*m*sizeof(int));
    nonMAX=(int*)malloc(n*m*sizeof(int));
    angle=(double*)malloc(n*m*sizeof(double));

    // Reading pixel data from file
    for (i = 0; i < n; i++)
    	for (j = 0; j < m; j++){
            index = j + i * m;
    		fscanf(inFile, "%d", &redSource[index]);
    		fscanf(inFile, "%d", &greenSource[index]);
    		fscanf(inFile, "%d", &blueSource[index]);
    	}

	stop1 = omp_get_wtime();
//----------------------------------------------------------------------
	start4 = omp_get_wtime();
	int mode=1000;
	while(mode!=1 && mode!=2 && mode!=3 && mode!=4 && mode!=5 && mode!=6&& mode!=7 && mode!=8)
	{
	printf("\nMode Selection\n \n1 copy	\n2 blur \n3 edge \n4 sharp \n5 noise\n6 graymap conversion\n7 canny edge detection\n8 gradient\n");
	scanf("%d",&mode);
	printf("\n");
	if(mode!=1 && mode!=2 && mode!=3 && mode!=4 && mode!=5 && mode!=6 && mode!=7 && mode!=8)
		{
			printf("\n\n\n\n\n\n");
			printf("-------------------------------------");
			printf("\nI'd rather you put in a proper number\n");
			printf("-------------------------------------\n");
		}}
	stop4 = omp_get_wtime();
	switch(mode)
	{	//copy
		case 1 :
			{start2 = omp_get_wtime();
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, copy);
			securite(redDest, greenDest, blueDest, n, m);    		
			stop2 = omp_get_wtime();}
		break;
		//blur
		case 2 :
			{start2 = omp_get_wtime();
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, blur);
			if((n*m)>(1920*1080))
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, blur);
			if((n*m)>(3840*2160))
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, blur);
			securite(redDest, greenDest, blueDest, n, m);    		
			stop2 = omp_get_wtime();}
		break;
		//simple edge detection
		case 3 :
			{start2 = omp_get_wtime();
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, edge);
			securite(redDest, greenDest, blueDest, n, m);    		
			stop2 = omp_get_wtime();}
		break;
		//basic sharpening
		case 4 :
			{start2 = omp_get_wtime();
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, sharp);
			if((n*m)>(1920*1080))
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, sharp);
			if((n*m)>(3840*2160))
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, sharp);
			securite(redDest, greenDest, blueDest, n, m);    		
			stop2 = omp_get_wtime();}
		break;
		//artificial noise generation
		case 5 :
			{start2 = omp_get_wtime();
			convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, noise);
			if((n*m)>(1920*1080))
				convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, noise);
			if((n*m)>(3840*2160))
				convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, noise);
			securite(redDest, greenDest, blueDest, n, m);    		
			stop2 = omp_get_wtime();}
		break;
		//black and white transformation
		case 6 :
			{
			for (i = 0; i < n; i++)
	    		for (j = 0; j < m; j++){
				index = j + i * m;
				nonMAX[index]=0;
	    		greyDest2[index] = 0.21*redSource[index]+0.72*greenSource[index]+0.07*blueSource[index];
	    		selector=1;
			}
			}
		break;
		//canny edge detection
		case 7 :
			{
				//thick selection
				puts("");
				puts("press 1 to thicken result (another number if not)");
				puts("useful for large picture (above 1920x1080)");
				scanf("%d",&thicker);
				//treshold selection
				puts("");
				puts("do you want treshold enabled");
				puts("0 for no  "); 
				puts("1 for yes ");
				puts("2 for auto teshold (for best results iterate std treshold)");
				while(selector!=1 && selector!=0 && selector!=2)
					{
						scanf("%d",&selector);
						if(selector!=1 && selector!=0 && selector!=2)
							{
									puts("-----------------------");
									puts("read instructions again");
									puts("-----------------------");
							}
					}
				if(selector==1)
				{
					puts("");
					puts("you now need to define treshold, it sets sensibility limit, the lower the more sensible");
					puts("");
					puts("define high treshold from 0 to 300");
					while(trH>301 | trH<0)
						{
							scanf("%d",&trH);
							if (trH>301 | trH<0)
								{
									puts("-----------------------");
									puts("read instructions again");
									puts("-----------------------");
								}
						}
					puts("define low treshold from 0 to 300");
					puts("remember that it needs to be equal or inferior to treshold high");
					while(trL>301 | trL<0 | trL>=trH)
						{
							scanf("%d",&trL);
							if (trL>301 | trL<0 | trL>=trH)
								{
									puts("-----------------------");
									puts("read instructions again");
									puts("-----------------------");
								}
						}
				}
				stop4 = omp_get_wtime();
				
				
				start2 = omp_get_wtime();
				
				
				//gauss convolution
				
				convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, gauss);
							
				//gradient creation			
							
				convolution_base(gradxR, gradxG, gradxB,
							redDest, greenDest, blueDest, n, m, gx);
				
				convolution_base(gradyR, gradyG, gradyB,
							redDest, greenDest, blueDest, n, m, gy);
							
				//gradient calculation	
					
				#pragma omp parallel for schedule(static, n)
				#pragma omp simd	
				for (i = 0; i < n; i++)
					for (j = 0; j < m; j++)
						{
							index = j + i * m;
							gradx[index]=0.21*gradxR[index]+0.72*gradxG[index]+0.07*gradxB[index];
							grady[index]=0.21*gradyR[index]+0.72*gradyG[index]+0.07*gradyB[index];
						}
				
				#pragma omp parallel for schedule(static, n)
				#pragma omp simd
				for (i = 0; i < n; i++)
					for (j = 0; j < m; j++)
						{
							index = j + i * m;
							edge_strenght[index]=sqrt(gradx[index]*gradx[index]+grady[index]*grady[index]);
						}
				//auto treshold calculation	
				if(selector==2)
					{
						#pragma omp parallel for schedule(static, n)
						#pragma omp simd
						for (i = 2; i < n-2; i++)
							for (j = 2; j < m-2; j++)
								{
									index = j + i * m;
									edge_avg+=edge_strenght[index];
								}
						edge_avg=edge_avg/((n-4)*(m-4));
						trH=0.8*edge_avg;
						trL=0.3*edge_avg;
					}
				//angle calculation
				#pragma omp parallel for schedule(static, n)
				#pragma omp simd
				for (i = 1; i < n-1; i++)
					for (j = 1; j < m-1; j++)
						{
							index = j + i * m;
							if (gradx[index]==0)
								{if(grady[index]==0)
									angle[index]=0;
								 else
									angle[index]=90;}
							else
								angle[index]=((atan(grady[index]/gradx[index]))*180)/PI;
							}
				//non maximum suppression
				
				#pragma omp parallel for schedule(static, n)
				#pragma omp simd
				for (i = 1; i < n-1; i++)
					for (j = 1; j < m-1; j++)
						{
							index = j + i * m;
							if(angle[index]<22.5  && angle[index]>=0)
								{
									if (edge_strenght[index]<edge_strenght[index+1] && edge_strenght[index]<edge_strenght[index-1])
										nonMAX[index]=255;
									if ((edge_strenght[index]>=edge_strenght[index+1] | edge_strenght[index]>=edge_strenght[index-1]))
										nonMAX[index]=0;
								}
							if(angle[index]<67.5  && angle[index]>=22.5)
								{
									if (edge_strenght[index]<edge_strenght[index+m+1] && edge_strenght[index]<edge_strenght[index-m-1])
										nonMAX[index]=255;
									if (edge_strenght[index]>=edge_strenght[index+m+1] | edge_strenght[index]>=edge_strenght[index-m-1])
										nonMAX[index]=0;
								}
							if(angle[index]<112.5 && angle[index]>=67.5)
								{
									if (edge_strenght[index]<edge_strenght[index-m] && edge_strenght[index]<edge_strenght[index+m])
										nonMAX[index]=255;
									if (edge_strenght[index]>=edge_strenght[index-m] | edge_strenght[index]>=edge_strenght[index+m])
										nonMAX[index]=0;
								}
							if(angle[index]<157.5 && angle[index]>=112.5)
								{
									if (edge_strenght[index]<edge_strenght[index-m+1] && edge_strenght[index]<edge_strenght[index+m-1])
										nonMAX[index]=255;
									if (edge_strenght[index]>=edge_strenght[index-m+1] | edge_strenght[index]>=edge_strenght[index+m-1])
										nonMAX[index]=0;
								}
							if(angle[index]<=180 && angle[index]>=157.5)
								{
									if (edge_strenght[index]<edge_strenght[index+1] && edge_strenght[index]<edge_strenght[index-1])
										nonMAX[index]=255;
									if (edge_strenght[index]>=edge_strenght[index+1] | edge_strenght[index]>=edge_strenght[index-1])
										nonMAX[index]=0;
								}
							}
							if(selector==1 | selector==2)
								{
									//treshold activation
									#pragma omp parallel for schedule(static, n)
									#pragma omp simd
									for (i = 1; i < n-1; i++)
										for (j = 1; j < m-1; j++)
											{
												index = j + i * m;
												if (edge_strenght[index]<trH)
													greyDest[index]=80;
												if (edge_strenght[index]<trL)
													greyDest[index]=0;
												if (edge_strenght[index]>=trH)
													greyDest[index]=255;
												if (nonMAX[index]==0)
													greyDest[index]=0;
											}
								}		
				//thicken image	
				if(thicker==1)	
					{
						convolution_BW(greyDest2, greyDest, n, m, thick);						
					}
				if(thicker!=1)
					convolution_BW(greyDest2, greyDest, n, m, copy);
				securite_BW(greyDest2, n, m);				
				
				stop2 = omp_get_wtime();
			}
		break;
		
		case 8 :
			{
				//treshold selection
				puts("");
				puts("do you want treshold enabled");
				puts("0 for no  "); 
				puts("1 for yes ");
				puts("2 for auto teshold (for best results iterate std treshold)");
				while(selector!=1 && selector!=0 && selector!=2)
					{
						scanf("%d",&selector);
						if(selector!=1 && selector!=0 && selector!=2)
							{
									puts("-----------------------");
									puts("read instructions again");
									puts("-----------------------");
							}
					}
				if(selector==1)
				{
					puts("");
					puts("you now need to define treshold, it sets sensibility limit, the lower the more sensible");
					puts("");
					puts("define high treshold from 0 to 300");
					while(trH>301 | trH<0)
						{
							scanf("%d",&trH);
							if (trH>301 | trH<0)
								{
									puts("-----------------------");
									puts("read instructions again");
									puts("-----------------------");
								}
						}
					puts("define low treshold from 0 to 300");
					puts("remember that it needs to be equal or inferior to treshold high");
					while(trL>301 | trL<0 | trL>=trH)
						{
							scanf("%d",&trL);
							if (trL>301 | trL<0 | trL>=trH)
								{
									puts("-----------------------");
									puts("read instructions again");
									puts("-----------------------");
								}
						}
				}
				stop4 = omp_get_wtime();
				
				
				start2 = omp_get_wtime();
				
				
				//gauss convolution
				
				convolution_base(redDest, greenDest, blueDest,
							redSource, greenSource, blueSource, n, m, gauss);
				
				//gradient creation
							
				convolution_base(gradxR, gradxG, gradxB,
							redDest, greenDest, blueDest, n, m, gx);
				
				convolution_base(gradyR, gradyG, gradyB,
							redDest, greenDest, blueDest, n, m, gy);
							
				//gradient calculation
							
				for (i = 0; i < n; i++)
					for (j = 0; j < m; j++)
						{
							index = j + i * m;
							gradx[index]=0.21*gradxR[index]+0.72*gradxG[index]+0.07*gradxB[index];
							grady[index]=0.21*gradyR[index]+0.72*gradyG[index]+0.07*gradyB[index];
							edge_strenght[index]=sqrt(gradx[index]*gradx[index]+grady[index]*grady[index]);
						}
				//auto treshold calculation	
				if(selector==2)
					{
						#pragma omp parallel for schedule(static, n)
						#pragma omp simd
						for (i = 2; i < n-2; i++)
							for (j = 2; j < m-2; j++)
								{
									index = j + i * m;
									edge_avg+=edge_strenght[index];
								}
						edge_avg=edge_avg/((n-4)*(m-4));
						trH=1.6*edge_avg;
						trL=0.9*edge_avg;
					}
					
				//treshold
				if(selector==1 | selector==2)
								{
									#pragma omp parallel for schedule(static, n)
									#pragma omp simd
									for (i = 1; i < n-1; i++)
										for (j = 1; j < m-1; j++)
											{
												index = j + i * m;
												if (edge_strenght[index]<trH)
													greyDest2[index]=80;
												if (edge_strenght[index]<trL)
													greyDest2[index]=0;
												if (edge_strenght[index]>=trH)
													greyDest2[index]=255;
											}
								}						
				if(selector==0)
					#pragma omp parallel for schedule(static, n)
					#pragma omp simd
					for (i = 1; i < n-1; i++)
										for (j = 1; j < m-1; j++)
											{
												index = j + i * m;
												nonMAX[index]=edge_strenght[index];
											}
				
				securite(redDest, greenDest, blueDest, n, m);  
				
				
				stop2 = omp_get_wtime();
			}
		break;
}
	
//----------------------------------------------------------------------
	start3 = omp_get_wtime();
    // Opening file
	outFile = fopen(argv[2], "w");
	if (outFile == NULL) {
	  printf("Can't open output file %s\n", argv[2]);
	  exit(1);
	}
	if(mode==6 | mode==7 | mode==8){
		// Writing metadata to file for graymap
		 fprintf(outFile, "P2\n");
    		 fprintf(outFile, "%d %d\n", m, n);
   		 fprintf(outFile, "%d\n", colorDepth);

		// Writing pixel data to file
		if(selector==0){
		 for (i = 0; i < n; i++){
    			for (j = 0; j < m; j++){
            			index = j + i * m;
            			fprintf(outFile, "%d ", nonMAX[index]);
    			}
    		 fprintf(outFile,"\n");
    		 }}
    	if(selector==1 | selector==2)
    	{
			for (i = 0; i < n; i++){
    			for (j = 0; j < m; j++){
            			index = j + i * m;
            			fprintf(outFile, "%d ", greyDest2[index]);
    			}
    		 fprintf(outFile,"\n");
    		 }	
			}
	}
	else
	{
    	// Writing metadata to file
		fprintf(outFile, "P3\n");
		fprintf(outFile, "%d %d\n", m, n);
		fprintf(outFile, "%d\n", colorDepth);

    	// Writing pixel data to file
		for (i = 0; i < n; i++){
		    	for (j = 0; j < m; j++){
		            index = j + i * m;
		            fprintf(outFile, "%d ", redDest[index]);
		            fprintf(outFile, "%d ", greenDest[index]);
		            fprintf(outFile, "%d ", blueDest[index]);
		    	}
		    	fprintf(outFile,"\n");
		}
     	}

	// Closing files
	fclose(inFile);
	fclose(outFile);

	// Deallocating memory
	free(redSource);
	free(greenSource);
	free(blueSource);
	free(redDest);
	free(greenDest);
	free(blueDest);
	free(greyDest);
	free(greyDest2);
	free(gradxR);
	free(gradyG);
	free(gradxB);
	free(gradyR);
	free(gradxG);
	free(gradyB);
	free(gradx);
	free(grady);
	free(edge_strenght);
	free(angle);
	free(nonMAX);

	stop3 = omp_get_wtime();
//----------------------------------------------------------------------
	time_fetch=stop1-start1;
	time_convolution =stop2-start2;
	time_writing=stop3-start3;
	time_thinking=stop4-start4;
	printf("\ntime fetch=%f s\ntime convolution=%f s\ntime writing=%f s\n\n",time_fetch,time_convolution,time_writing);
	printf("almost forgot, it took you %f seconds to know what you wanted me to do\n\n",time_thinking);
	
	return 0;
}

