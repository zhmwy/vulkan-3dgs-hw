

void sort_small(inout float d[32], int n)
{
    int j;
    for (int i = 1; i < n; i++)
    {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
        {
            d[j] = d[j-1];
        }
        d[j] = tmp;
    }
}