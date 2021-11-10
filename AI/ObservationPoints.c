#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OB_LIST_N 6740
#define MESH_SIZE 256

#define JAPAN_LAT_S 10.0 
#define JAPAN_LAT_N 46.0

#define JAPAN_LON_E 154.0
#define JAPAN_LON_W 122.0

typedef struct{
    int ob_p_ID;
    //char ob_p_name[];
    double latitude;
    double longitude;
    //int ob_p_start;
} OB_LIST;

int latitude2Ycoor(double latitude);
int longitude2Xcoor(double longitude);

int main(void){
    char filename_list[] = "data_original/code_p.dat";
    char filename_out[] = "data_shaped/ObservationPointsMap.csv";
    FILE  *fp_list, *fp_out;

    OB_LIST ob_list[OB_LIST_N];
    int ob_p_ID;
    double latitude, longitude;
    char ob_p_name[100];

    int mesh[MESH_SIZE][MESH_SIZE] = {0};
    int x, y;

    if ((fp_list = fopen(filename_list, "r")) == NULL){
        printf("cannot open file:%s\n", filename_list);
        return 1;
    }
    if ((fp_out = fopen(filename_out, "w")) == NULL){
        printf("cannot open file:%s\n", filename_out);
        return 1;
    }

    for (int i = 0; i < OB_LIST_N; i++){
        if(EOF == fscanf(fp_list, "%d %s %lf %lf%*[^\n]%*[\n]", &ob_p_ID, ob_p_name, &latitude, &longitude)){
            printf("maybe faild to load observation point list\n");
            break;
        }
        ob_list[i].ob_p_ID = ob_p_ID;
        //ob_list[i].p_ = ob_p_name;
        ob_list[i].latitude = latitude / 100.0;
        ob_list[i].longitude = longitude / 100.0;
        x = longitude2Xcoor(ob_list[i].longitude);
        y = latitude2Ycoor(ob_list[i].latitude);
        if (0 <= x && x < 256 && 0 <= y && y < 256){
            mesh[x][y] ++;
        }
        else{
            printf("座標 x, y = %d, %d, ID = %d, %s\n", x, y, ob_p_ID, ob_p_name);
        }
    }
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
    fclose(fp_list);
    fclose(fp_out);
    return 0;
}

int latitude2Ycoor(double latitude){
    int latitude_deg = latitude;
    double latitude_min = latitude - latitude_deg;
    double latitude_in_100 = latitude_deg + (latitude_min / 60.0) * 100.0;
    double lat_s = JAPAN_LAT_S;
    double lat_n = JAPAN_LAT_N;
    double lat_width;
    int MeshSize = MESH_SIZE;
    int y; //左上が 0 

    lat_width = lat_n - lat_s;
    //printf("latitude width %f\n", MeshSize * (latitude - lat_s) / lat_width);

    y = MeshSize * (latitude_in_100 - lat_s) / lat_width;

    return MeshSize - y;
}
int longitude2Xcoor(double longitude){
    int longitude_deg = longitude;
    double longitude_min = longitude - longitude_deg;
    double longitude_in_100 = longitude_deg + (longitude_min / 60.0) * 100.0;
    double lon_w = JAPAN_LON_W;
    double lon_e = JAPAN_LON_E;
    double lon_width;
    int MeshSize = MESH_SIZE;
    int x;  //左上が 0 

    lon_width = lon_e - lon_w;
    //printf("longitude %f, x %f\n", longitude, (longitude - lon_w) / lon_width * MeshSize);

    x = (longitude_in_100 - lon_w) / lon_width * MeshSize;

    return x;
}