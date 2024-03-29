#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OB_LIST_N 6740
#define YEAR_S 1997 //1997
#define YEAR_E 2019 //2017
#define MESH_SIZE 256

#define JAPAN_LAT_S 10.0    //10.0
#define JAPAN_LAT_N 46.0    //46.0

#define JAPAN_LON_E 154.0   //154.0
#define JAPAN_LON_W 122.0   //122.0

typedef struct{
    int EarthQuake_ID;
    int ob_n;
    double latitude;
    double longitude;
    double depth;
    double magnitude;
} Epicenter;

typedef struct{
    int EarthQuake_ID;

    double latitude;
    double longitude;

    double SeismicIntensity;
    int IntensityClass; //0 ~ 9

} OB_DATA;

int latitude2Ycoor(double latitude);
int longitude2Xcoor(double longitude);
double degmin2_100(double tude);
int int_max(int a, int b);

int main(void){
    Epicenter epic;
    OB_DATA ob_data;


    for (int i = YEAR_S; i <= YEAR_E; i++){
        char filename_in_epic[100] = "Earthquake_Data/data_shaped/shingen";    //.csv
        char filename_in_ob_data[100] = "Earthquake_Data/data_shaped/kansoku"; //.csv
        FILE *fp_in_epic, *fp_in_ob_data;

        char file_year_str[100];
        sprintf(file_year_str, "%d", i); 

        strcat(filename_in_epic, file_year_str);
        strcat(filename_in_epic, ".csv");
        strcat(filename_in_ob_data, file_year_str);
        strcat(filename_in_ob_data, ".csv");
        printf("input : %s, %s\n", filename_in_epic, filename_in_ob_data);
        if ((fp_in_epic = fopen(filename_in_epic, "r")) == NULL){
            printf("cannot open file:%s\n", filename_in_epic);
            return 1;
        }
        if ((fp_in_ob_data = fopen(filename_in_ob_data, "r")) == NULL){
            printf("cannot open file:%s\n", filename_in_ob_data);
            return 1;
        }
        
        while(EOF != fscanf(fp_in_epic, "%d,%d,%lf,%lf,%lf,%lf", &epic.EarthQuake_ID, &epic.ob_n, &epic.latitude, &epic.longitude, &epic.depth, &epic.magnitude)){
            char filename_out[100] = "Earthquake_Data/data_reshaped/";  //.csv
            FILE *fp_out;
            char EarthQuake_ID_str[100];
            sprintf(EarthQuake_ID_str, "%d", epic.EarthQuake_ID);
            strcat(filename_out, EarthQuake_ID_str);
            strcat(filename_out, ".csv");
            //printf("output:%s\n", filename_out);

            int mesh[MESH_SIZE][MESH_SIZE] = {0};
            int epic_x, epic_y;
            epic_x = longitude2Xcoor(epic.longitude);
            epic_y = latitude2Ycoor(epic.latitude);

            for (int j = 0; j < epic.ob_n; j++){
                if(5 == fscanf(fp_in_ob_data, "%d,%lf,%lf,%lf,%d", &ob_data.EarthQuake_ID, &ob_data.latitude, &ob_data.longitude, &ob_data.SeismicIntensity, &ob_data.IntensityClass)){
                    int y = latitude2Ycoor(ob_data.latitude);
                    int x = longitude2Xcoor(ob_data.longitude);
                    if(0 <= x && x < MESH_SIZE && 0 <= y && y < MESH_SIZE){
                        mesh[x][y] = int_max(ob_data.IntensityClass, mesh[x][y]);
                    }
                }
            }

            if(0 <= epic_x && epic_x < MESH_SIZE && 0 <= epic_y && epic_y < MESH_SIZE){
                if ((fp_out = fopen(filename_out, "w")) == NULL){
                    printf("cannot open file:%s\n", filename_out);
                    return 1;
                }
                fprintf(fp_out, "%d,%d,%f,%f\n", epic_x, epic_y, epic.depth, epic.magnitude);
                for (int y = 0; y < MESH_SIZE; y++){
                    for (int x = 0; x < MESH_SIZE; x++){
                        if (x  < MESH_SIZE - 1){
                            fprintf(fp_out, "%d,", mesh[x][y]);
                        }
                        else{
                            fprintf(fp_out, "%d\n", mesh[x][y]);
                        }
                    }
                }
                fclose(fp_out);
            }
        }
        fclose(fp_in_epic);
        fclose(fp_in_ob_data);
    }
}

int latitude2Ycoor(double latitude){
    double latitude_in_100 = degmin2_100(latitude);
    double lat_s = JAPAN_LAT_S;
    double lat_n = JAPAN_LAT_N;
    double lat_s_in_100 = degmin2_100(lat_s);
    double lat_n_in_100 = degmin2_100(lat_n);
    double lat_width_in_100;
    int MeshSize = MESH_SIZE;
    int y; //左上が 0 

    lat_width_in_100 = lat_n_in_100 - lat_s_in_100;

    y = MeshSize * (latitude_in_100 - lat_s_in_100) / lat_width_in_100;

    return MeshSize - y;
}
int longitude2Xcoor(double longitude){
    double longitude_in_100 = degmin2_100(longitude);
    double lon_w = JAPAN_LON_W;
    double lon_e = JAPAN_LON_E;
    double lon_w_in_100 = degmin2_100(lon_w);
    double lon_e_in_100 = degmin2_100(lon_e);
    double lon_width_in_100;
    int MeshSize = MESH_SIZE;
    int x;  //左上が 0 

    lon_width_in_100 = lon_e_in_100 - lon_w_in_100;

    x = MeshSize * (longitude_in_100 - lon_w_in_100) / lon_width_in_100;

    return x;
}

double degmin2_100(double tude){
    int tude_deg = tude;
    double tude_min = tude - tude_deg;
    double tude_in_100 = tude_deg + (tude_min / 60.0) * 100;

    return tude_in_100;
}

int int_max(int a, int b){
    if (a > b){
        return a;
    }else{
        return b;
    }
}