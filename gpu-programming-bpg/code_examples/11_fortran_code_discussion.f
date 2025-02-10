k2 = 0
do i = 1, n_sites
    do j = 1, n_neigh(i)
        k2 = k2 + 1
        counter = 0
        counter2 = 0
        do n = 1, n_max
            do np = n, n_max
                do l = 0, l_max
                    if( skip_soap_component(l, np, n) )cycle

                    counter = counter+1
                    do m = 0, l
                        k = 1 + l*(l+1)/2 + m
                        counter2 = counter2 + 1
                        multiplicity = multiplicity_array(counter2)
                        soap_rad_der(counter, k2) = soap_rad_der(counter, k2) + multiplicity * real( cnk_rad_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_rad_der(k, np, k2)) )
                        soap_azi_der(counter, k2) = soap_azi_der(counter, k2) + multiplicity * real( cnk_azi_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_azi_der(k, np, k2)) )
                        soap_pol_der(counter, k2) = soap_pol_der(counter, k2) + multiplicity * real( cnk_pol_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_pol_der(k, np, k2)) )
                    end do
                end do
            end do
        end do

        soap_rad_der(1:n_soap, k2) = soap_rad_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_rad_der(1:n_soap, k2) )
        soap_azi_der(1:n_soap, k2) = soap_azi_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_azi_der(1:n_soap, k2) )
        soap_pol_der(1:n_soap, k2) = soap_pol_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_pol_der(1:n_soap, k2) )

        if( j == 1 )then
            k3 = k2
        else
            soap_cart_der(1, 1:n_soap, k2) = dsin(thetas(k2)) * dcos(phis(k2)) * soap_rad_der(1:n_soap, k2) - dcos(thetas(k2)) * dcos(phis(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2) - dsin(phis(k2)) / rjs(k2) * soap_azi_der(1:n_soap, k2)
            soap_cart_der(2, 1:n_soap, k2) = dsin(thetas(k2)) * dsin(phis(k2)) * soap_rad_der(1:n_soap, k2) - dcos(thetas(k2)) * dsin(phis(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2) + dcos(phis(k2)) / rjs(k2) * soap_azi_der(1:n_soap, k2)
            soap_cart_der(3, 1:n_soap, k2) = dcos(thetas(k2)) * soap_rad_der(1:n_soap, k2) + dsin(thetas(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2)
            soap_cart_der(1, 1:n_soap, k3) = soap_cart_der(1, 1:n_soap, k3) - soap_cart_der(1, 1:n_soap, k2)
            soap_cart_der(2, 1:n_soap, k3) = soap_cart_der(2, 1:n_soap, k3) - soap_cart_der(2, 1:n_soap, k2)
            soap_cart_der(3, 1:n_soap, k3) = soap_cart_der(3, 1:n_soap, k3) - soap_cart_der(3, 1:n_soap, k2)
        end if
    end do
end do
