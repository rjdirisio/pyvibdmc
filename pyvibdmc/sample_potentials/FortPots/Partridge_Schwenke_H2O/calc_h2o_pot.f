        SUBROUTINE calc_hoh_pot(x, nwalk, v)
        implicit real*8 (a-h,o-z)

        parameter (natom = 3)
        parameter (ndim = 3*natom)
        integer, intent(in) :: nwalk
        double precision, dimension(nwalk,natom,3), intent(in) :: x
        double precision, dimension (nwalk), intent(out) :: v

        dimension rij(nwalk,3)

        do k = 1,nwalk

           r1 = 0.
           r2 = 0.
           ct = 0.
           do j = 1,3
              d1 = x(k,3,j)-x(k,1,j)
              d2 = x(k,3,j)-x(k,2,j)
              r1 = r1 + d1**2
              r2 = r2 + d2**2
              ct = ct + d1*d2
           enddo
           rij(k,1) = sqrt(r1)
           rij(k,2) = sqrt(r2)
           rij(k,3) = acos(CT/rij(k,1)/rij(k,2))


        enddo

        call vibpot(rij,v,nwalk)

        
        end
        
