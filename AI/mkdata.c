#include <stdio.h>

#define OB_LIST_N 6740

typedef struct{
    int p_ID;
    //char ob_p_name[];
    double latitude;
    double longitude;
    int p_start;
} OB_LIST;

int main(void){
    char filename_data = "data/i2019.dat";
    char filename_list = "data/format_j.dat";
    char filename_out = "data/data.dat";
    FILE *fp_data, *fp_list, *fp_out;

    //観測点リスト用
    OB_LIST ob_list[OB_LIST_N];
    int p_ID, p_start;
    char ob_p_name[100];
    double latitude, longitude;

    //データ用
    char head;
    int year, month, day, hour, min, sec;
    double depth;

    if (fp_data = fopen(filename_data, "r") == NULL){
        printf("cannot open file\n");
    }
    if (fp_list = fopen(filename_list, "r") == NULL){
        printf("cannot open file\n");
    }
    if (fp_out = fopen(filename_out, "w") == NULL){
        printf("cannot open file\n");
    }

    for (int i = 0; i < OB_LIST_N; i++){
        if(EOF == fscanf(fp_list, "%d %s %lf %lf %d\n", &p_ID, ob_p_name, &latitude, &longitude, &p_start)){
            prinf("maybe faild to load observation point list\n");
            break;
        }
        ob_list[i].p_ID = p_ID;
        //ob_list[i].p_ = ob_p_name;
        ob_list[i].latitude = latitude / 100.0;
        ob_list[i].longitude = longitude / 100.0;
        ob_list[i].p_start = p_start;
    }

    while (head = fgetc(fp_data) != NULL){
        if (head == 'A'){
            fscanf("%4d%2d%2d%2d%2d %*d %lf %*d %lf %*d %4lf%*3d%lf%*c ", &year, &month, &day, &hour, &min, &sec, &latitude, &longitude, &depth, &Magnitude);
            fscanf("%[]s");
        }
    }

}