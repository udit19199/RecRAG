Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=9d446d266e419f87 ===
Gold sentences and facts:
Sentence 0: Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France `` for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays '' 1906 Joseph John Thomson United Kingdom `` for his theoretical and experimental investigations on the conduction of electricity by gases '' 1907 Albert Abraham Michelson United States Poland `` for his optical precision instruments and the spectroscopic and metrological investigations carried out with their aid '' 1908 Gabriel Lippmann France `` for his method of reproducing colours photographically based on the phenomenon of interference '' 1909 Guglielmo Marconi Italy `` for their contributions to the development of wireless telegraphy '' Karl Ferdinand Braun Germany 1910 Johannes Diderik van der Waals Netherlands `` for his work on the equation of state for gases and liquids '' 1911 Wilhelm Wien Germany `` for his discoveries regarding the laws governing the radiation of heat '' 1912 Nils Gustaf Dalén Sweden `` for his invention of automatic valves designed to be used in combination with gas accumulators in lighthouses and buoys '' 1913 Heike Kamerlingh - Onnes Netherlands `` for his investigations on the properties of matter at low temperatures which led , inter alia , to the production of liquid helium '' 1914 Max von Laue Germany `` For his discovery of the diffraction of X-rays by crystals '' , an important step in the development of X-ray spectroscopy . 1915 William Henry Bragg United Kingdom `` For their services in the analysis of crystal structure by means of X-rays '' , an important step in the development of X-ray crystallography William Lawrence Bragg Australia United Kingdom 1916 Not awarded World War I 1917 Charles Glover Barkla United Kingdom `` For his discovery of the characteristic Röntgen radiation of the elements '' , another important step in the development of X-ray spectroscopy 1918 Max Planck Germany `` for the services he rendered to the advancement of physics by his discovery of energy quanta '' 1919 Johannes Stark Germany `` for his discovery of the Doppler effect in canal rays and the splitting of spectral lines in electric fields '' 1920 Charles Édouard Guillaume Switzerland `` for the service he has rendered to precision measurements in physics by his discovery of anomalies in nickel - steel alloys '' 1921 Albert Einstein Germany Switzerland `` for his services to theoretical physics , and especially for his discovery of the law of the photoelectric effect '' 1922 Niels Bohr Denmark `` for his services in the investigation of the structure of atoms and of the radiation emanating from them '' 1923 Robert Andrews Millikan United States `` for his work on the elementary charge of electricity and on the photoelectric effect '' 1924 Manne Siegbahn Sweden `` for his discoveries and research in the field of X-ray spectroscopy '' 1925 James Franck Germany `` for their discovery of the laws governing the impact of an electron upon an atom '' Gustav Hertz Germany 1926 Jean Baptiste Perrin France `` for his work on the discontinuous structure of matter , and especially for his discovery of sedimentation equilibrium '' 1927 Arthur Holly Compton United States `` for his discovery of the effect named after him '' Charles Thomson Rees Wilson United Kingdom `` for his method of making the paths of electrically charged particles visible by condensation of vapour '' 1928 Owen Willans Richardson United Kingdom `` for his work on the thermionic phenomenon and especially for the discovery of the law named after him '' 1929 Louis Victor Pierre Raymond , 7th Duc de Broglie France `` for his discovery of the wave nature of electrons '' 1930 Chandrasekhara Venkata Raman India `` for his work on the scattering of light and for the discovery of the effect named after him '' 1931 Not awarded 1932 Werner Heisenberg Germany `` for the creation of quantum mechanics , the application of which has , inter alia , led to the discovery of the allotropic forms of hydrogen '' 1933 Erwin Schrödinger Austria `` for the discovery of new productive forms of atomic theory '' Paul Dirac United Kingdom 1934 Not awarded 1935 James Chadwick United Kingdom `` for the discovery of the neutron '' 1936 Victor Francis Hess Austria `` for his discovery of cosmic radiation '' Carl David Anderson United States `` for his discovery of the positron '' 1937 Clinton Joseph Davisson United States `` for their experimental discovery of the diffraction of electrons by crystals '' George Paget Thomson United Kingdom 1938 Enrico Fermi Italy `` for his demonstrations of the existence of new radioactive elements produced by neutron irradiation , and for his related discovery of nuclear reactions brought about by slow neutrons '' 1939 Ernest Lawrence United States `` for the invention and development of the cyclotron and for results obtained with it , especially with regard to artificial radioactive elements '' 1940 Not awarded World War II 1941 Not awarded World War II 1942 Not awarded World War II 1943 Otto Stern United States `` for his contribution to the development of the molecular ray method and his discovery of the magnetic moment of the proton '' 1944 Isidor Isaac Rabi United States Poland `` for his resonance method for recording the magnetic properties of atomic nuclei '' 1945 Wolfgang Pauli Austria `` for the discovery of the Exclusion Principle , also called the Pauli principle '' 1946 Percy Williams Bridgman United States `` for the invention of an apparatus to produce extremely high pressures , and for the discoveries he made there within the field of high pressure physics '' 1947 Edward Victor Appleton United Kingdom `` for his investigations of the physics of the upper atmosphere especially for the discovery of the so - called Appleton layer '' 1948 Patrick Maynard Stuart Blackett United Kingdom `` for his development of the Wilson cloud chamber method , and his discoveries therewith in the fields of nuclear physics and cosmic radiation '' 1949 Hideki Yukawa Japan `` for his prediction of the existence of mesons on the basis of theoretical work on nuclear forces '' 1950 Cecil Frank Powell United Kingdom `` for his development of the photographic method of studying nuclear processes and his discoveries regarding mesons made with this method '' 1951 John Douglas Cockcroft United Kingdom `` for their pioneer work on the transmutation of atomic nuclei by artificially accelerated atomic particles '' Ernest Thomas Sinton Walton Ireland 1952 Felix Bloch Switzerland United States `` for their development of new methods for nuclear magnetic precision measurements and discoveries in connection therewith '' Edward Mills Purcell United States 1953 Frits Zernike Netherlands `` for his demonstration of the phase contrast method , especially for his invention of the phase contrast microscope '' 1954 Max Born West Germany `` for his fundamental research in quantum mechanics , especially for his statistical interpretation of the wavefunction '' Walther Bothe West Germany `` for the coincidence method and his discoveries made therewith '' 1955 Willis Eugene Lamb United States `` for his discoveries concerning the fine structure of the hydrogen spectrum '' Polykarp Kusch United States Germany `` for his precision determination of the magnetic moment of the electron '' 1956 John Bardeen United States `` for their researches on semiconductors and their discovery of the transistor effect '' Walter Houser Brattain United States William Bradford Shockley United States 1957 Tsung - Dao Lee Republic of China `` for their penetrating investigation of the so - called parity laws which has led to important discoveries regarding the elementary particles '' Chen Ning Yang Republic of China 1958 Pavel Alekseyevich Cherenkov Soviet Union `` for the discovery and the interpretation of the Cherenkov effect '' Ilya Frank Soviet Union Igor Yevgenyevich Tamm Soviet Union 1959 Emilio Gino Segrè Italy United States `` for their discovery of the antiproton '' Owen Chamberlain United States 1960 Donald Arthur Glaser United States `` for the invention of the bubble chamber '' 1961 Robert Hofstadter United States `` for his pioneering studies of electron scattering in atomic nuclei and for his thereby achieved discoveries concerning the structure of the nucleons '' Rudolf Ludwig Mössbauer West Germany `` for his researches concerning the resonance absorption of gamma radiation and his discovery in this connection of the effect which bears his name '' 1962 Lev Davidovich Landau Soviet Union `` for his pioneering theories for condensed matter , especially liquid helium '' 1963 Eugene Paul Wigner Hungary United States `` for his contributions to the theory of the atomic nucleus and the elementary particles , particularly through the discovery and application of fundamental symmetry principles '' Maria Goeppert - Mayer United States `` for their discoveries concerning nuclear shell structure '' J. Hans D. Jensen West Germany 1964 Nicolay Gennadiyevich Basov Soviet Union `` for fundamental work in the field of quantum electronics , which has led to the construction of oscillators and amplifiers based on the maser -- laser principle '' Alexander Prokhorov Soviet Union Charles Hard Townes United States 1965 Richard Phillips Feynman United States `` for their fundamental work in quantum electrodynamics ( QED ) , with deep - ploughing consequences for the physics of elementary particles '' Julian Schwinger United States Shin'ichirō Tomonaga Japan 1966 Alfred Kastler France `` for the discovery and development of optical methods for studying Hertzian resonances in atoms '' 1967 Hans Albrecht Bethe United States `` for his contributions to the theory of nuclear reactions , especially his discoveries concerning the energy production in stars '' 1968 Luis Walter Alvarez United States `` for his decisive contributions to elementary particle physics , in particular the discovery of a large number of resonance states , made possible through his development of the technique of using hydrogen bubble chamber and data analysis '' 1969 Murray Gell - Mann United States `` for his contributions and discoveries concerning the classification of elementary particles and their interactions '' 1970 Hannes Olof Gösta Alfvén Sweden `` for fundamental work and discoveries in magneto - hydrodynamics with fruitful applications in different parts of plasma physics '' Louis Néel France `` for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism which have led to important applications in solid state physics '' 1971 Dennis Gabor Hungary United Kingdom `` for his invention and development of the holographic method '' 1972 John Bardeen United States `` for their jointly developed theory of superconductivity , usually called the BCS - theory '' Leon Neil Cooper United States John Robert Schrieffer United States 1973 Leo Esaki Japan `` for their experimental discoveries regarding tunneling phenomena in semiconductors and superconductors , respectively '' Ivar Giaever United States Norway Brian David Josephson United Kingdom `` for his theoretical predictions of the properties of a supercurrent through a tunnel barrier , in particular those phenomena which are generally known as the Josephson effect '' Martin Ryle United Kingdom `` for their pioneering research in radio astrophysics : Ryle for his observations and inventions , in particular of the aperture synthesis technique , and Hewish for his decisive role in the discovery of pulsars '' Antony Hewish United Kingdom Aage Bohr Denmark `` for the discovery of the connection between collective motion and particle motion in atomic nuclei and the development of the theory of the structure of the atomic nucleus based on this connection '' Ben Roy Mottelson Denmark Leo James Rainwater United States 1976 Burton Richter United States `` for their pioneering work in the discovery of a heavy elementary particle of a new kind '' Samuel Chao Chung Ting United States 1977 Philip Warren Anderson United States `` for their fundamental theoretical investigations of the electronic structure of magnetic and disordered systems '' Nevill Francis Mott United Kingdom John Hasbrouck Van Vleck United States 1978 Pyotr Leonidovich Kapitsa Soviet Union `` for his basic inventions and discoveries in the area of low - temperature physics '' Arno Allan Penzias United States `` for their discovery of cosmic microwave background radiation '' Robert Woodrow Wilson United States 1979 Sheldon Lee Glashow United States `` for their contributions to the theory of the unified weak and electromagnetic interaction between elementary particles , including , inter alia , the prediction of the weak neutral current '' Abdus Salam Pakistan Steven Weinberg United States 1980 James Watson Cronin United States `` for the discovery of violations of fundamental symmetry principles in the decay of neutral K - mesons '' Val Logsdon Fitch United States 1981 Nicolaas Bloembergen Netherlands United States `` for their contribution to the development of laser spectroscopy '' Arthur Leonard Schawlow United States Kai Manne Börje Siegbahn Sweden `` for his contribution to the development of high - resolution electron spectroscopy '' 1982 Kenneth G. Wilson United States `` for his theory for critical phenomena in connection with phase transitions '' Subrahmanyan Chandrasekhar India United States `` for his theoretical studies of the physical processes of importance to the structure and evolution of the stars '' William Alfred Fowler United States `` for his theoretical and experimental studies of the nuclear reactions of importance in the formation of the chemical elements in the universe '' 1984 Carlo Rubbia Italy `` for their decisive contributions to the large project , which led to the discovery of the field particles W and Z , communicators of weak interaction '' Simon van der Meer Netherlands 1985 Klaus von Klitzing West Germany `` for the discovery of the quantized Hall effect '' 1986 Ernst Ruska West Germany `` for his fundamental work in electron optics , and for the design of the first electron microscope '' Gerd Binnig West Germany `` for their design of the scanning tunneling microscope '' Heinrich Rohrer Switzerland Johannes Georg Bednorz West Germany `` for their important break - through in the discovery of superconductivity in ceramic materials '' Karl Alexander Müller Switzerland 1988 Leon Max Lederman United States `` for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino '' Melvin Schwartz United States Jack Steinberger United States 1989 Norman Foster Ramsey United States `` for the invention of the separated oscillatory fields method and its use in the hydrogen maser and other atomic clocks '' Hans Georg Dehmelt United States Germany `` for the development of the ion trap technique '' Wolfgang Paul West Germany 1990 Jerome I. Friedman United States `` for their pioneering investigations concerning deep inelastic scattering of electrons on protons and bound neutrons , which have been of essential importance for the development of the quark model in particle physics '' Henry Way Kendall United States Richard E. Taylor Canada 1991 Pierre - Gilles de Gennes France `` for discovering that methods developed for studying order phenomena in simple systems can be generalized to more complex forms of matter , in particular to liquid crystals and polymers '' 1992 Georges Charpak France Poland `` for his invention and development of particle detectors , in particular the multiwire proportional chamber '' 1993 Russell Alan Hulse United States `` for the discovery of a new type of pulsar , a discovery that has opened up new possibilities for the study of gravitation '' Joseph Hooton Taylor , Jr . United States 1994 Bertram Brockhouse Canada `` for the development of neutron spectroscopy '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' Clifford Glenwood Shull United States `` for the development of the neutron diffraction technique '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' 1995 Martin Lewis Perl United States `` for the discovery of the tau lepton '' and `` for pioneering experimental contributions to lepton physics '' Frederick Reines United States `` for the detection of the neutrino '' and `` for pioneering experimental contributions to lepton physics '' David Morris Lee United States `` for their discovery of superfluidity in helium - 3 '' Douglas D. Osheroff United States Robert Coleman Richardson United States 1997 Steven Chu United States `` for development of methods to cool and trap atoms with laser light . '' Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands `` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002 Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics , in particular for the detection of cosmic neutrinos '' Masatoshi Koshiba Japan Riccardo Giacconi Italy United States `` for pioneering contributions to astrophysics , which have led to the discovery of cosmic X-ray sources '' 2003 Alexei Alexeyevich Abrikosov Russia United States `` for pioneering contributions to the theory of superconductors and superfluids '' Vitaly Lazarevich Ginzburg Russia Anthony James Leggett United Kingdom United States David J. Gross United States `` for the discovery of asymptotic freedom in the theory of the strong interaction '' Hugh David Politzer United States Frank Wilczek United States 2005 Roy J. Glauber United States `` for his contribution to the quantum theory of optical coherence '' John L. Hall United States `` for their contributions to the development of laser - based precision spectroscopy , including the optical frequency comb technique '' Theodor W. Hänsch Germany 2006 John C. Mather United States `` for their discovery of the blackbody form and anisotropy of the cosmic microwave background radiation '' George F. Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and manipulation of individual quantum systems . '' David J. Wineland United States 2013 François Englert Belgium `` for the theoretical discovery of a mechanism that contributes to our understanding of the origin of mass of subatomic particles , and which recently was confirmed through the discovery of the predicted fundamental particle , by the ATLAS and CMS experiments at CERN 's Large Hadron Collider '' Peter Higgs United Kingdom 2014 Isamu Akasaki Japan `` for the invention of efficient blue light - emitting diodes which has enabled bright and energy - saving white light sources '' Hiroshi Amano Japan Shuji Nakamura Japan United States 2015 Takaaki Kajita Japan `` for the discovery of neutrino oscillations , which shows that neutrinos have mass '' Arthur B. McDonald Canada 2016 David J. Thouless United Kingdom United States `` for theoretical discoveries of topological phase transitions and topological phases of matter '' F. Duncan M. Haldane United Kingdom United States John M. Kosterlitz United Kingdom United States 2017 Rainer Weiss Germany United States `` for decisive contributions to the LIGO detector and the observation of gravitational waves '' Kip Thorne United States Barry Barish United States
  Fact 0.0: 1901: Wilhelm Conrad Röntgen of Germany won the physics prize for the discovery of the remarkable rays later named after him
  Fact 0.1: 1902: Hendrik Lorentz of the Netherlands won the physics prize for research into the influence of magnetism upon radiation phenomena
  Fact 0.2: 1902: Pieter Zeeman of the Netherlands shared the physics prize with Hendrik Lorentz
  Fact 0.3: 1903: Antoine Henri Becquerel of France won the physics prize for the discovery of spontaneous radioactivity
  Fact 0.4: 1903: Pierre Curie of France shared the physics prize for joint research on the radiation phenomena discovered by Henri Becquerel
  Fact 0.5: 1903: Maria Skłodowska-Curie, listed as Poland and France, shared the physics prize for the same joint research on radiation phenomena
  Fact 0.6: 1904: Lord Rayleigh of the United Kingdom won the physics prize for investigating the densities of the most important gases
  Fact 0.7: 1904: Lord Rayleigh of the United Kingdom won the physics prize for discovering argon
  Fact 0.8: 1905: Philipp Eduard Anton von Lenard, listed as Austria-Hungary and Germany, won the physics prize for his work on cathode rays
  Fact 0.9: 1906: Joseph John Thomson of the United Kingdom won the physics prize for theoretical and experimental investigations of electricity conduction by gases
  Fact 0.10: 1907: Albert Abraham Michelson, listed as the United States and Poland, won the physics prize for optical precision instruments
  Fact 0.11: 1907: Albert Abraham Michelson won the physics prize for spectroscopic and metrological investigations carried out with those instruments
  Fact 0.12: 1908: Gabriel Lippmann of France won the physics prize for a method of reproducing colours photographically based on interference
  Fact 0.13: 1909: Guglielmo Marconi of Italy won the physics prize for contributions to the development of wireless telegraphy
  Fact 0.14: 1909: Karl Ferdinand Braun of Germany shared the physics prize for contributions to wireless telegraphy
  Fact 0.15: 1910: Johannes Diderik van der Waals of the Netherlands won the physics prize for work on the equation of state for gases and liquids
  Fact 0.16: 1911: Wilhelm Wien of Germany won the physics prize for discoveries regarding the laws governing the radiation of heat
  Fact 0.17: 1912: Nils Gustaf Dalén of Sweden won the physics prize for inventing automatic valves used with gas accumulators in lighthouses and buoys
  Fact 0.18: 1913: Heike Kamerlingh-Onnes of the Netherlands won the physics prize for investigations of matter at low temperatures, which led to the production of liquid helium
  Fact 0.19: 1914: Max von Laue of Germany won the physics prize for discovering the diffraction of X-rays by crystals
  Fact 0.20: 1915: William Henry Bragg of the United Kingdom shared the physics prize for analysing crystal structure by means of X-rays
  Fact 0.21: 1915: William Lawrence Bragg, listed as Australia and the United Kingdom, shared the physics prize for analysing crystal structure by means of X-rays
  Fact 0.22: 1916: No physics prize was awarded because of World War I
  Fact 0.23: 1917: Charles Glover Barkla of the United Kingdom won the physics prize for discovering the characteristic Röntgen radiation of the elements
  Fact 0.24: 1918: Max Planck of Germany won the physics prize for services to physics through his discovery of energy quanta
  Fact 0.25: 1919: Johannes Stark of Germany won the physics prize for discovering the Doppler effect in canal rays and the splitting of spectral lines in electric fields
  Fact 0.26: 1920: Charles Édouard Guillaume of Switzerland won the physics prize for precision measurements in physics through his discovery of anomalies in nickel-steel alloys
  Fact 0.27: 1921: Albert Einstein, listed as Germany and Switzerland, won the physics prize for services to theoretical physics
  Fact 0.28: 1921: Albert Einstein won the physics prize especially for discovering the law of the photoelectric effect
  Fact 0.29: 1922: Niels Bohr of Denmark won the physics prize for investigating the structure of atoms and the radiation emanating from them
  Fact 0.30: 1923: Robert Andrews Millikan of the United States won the physics prize for work on the elementary charge of electricity and on the photoelectric effect
  Fact 0.31: 1924: Manne Siegbahn of Sweden won the physics prize for discoveries and research in X-ray spectroscopy
  Fact 0.32: 1925: James Franck of Germany won the physics prize for discovering the laws governing the impact of an electron upon an atom
  Fact 0.33: 1925: Gustav Hertz of Germany shared the physics prize with James Franck
  Fact 0.34: 1926: Jean Baptiste Perrin of France won the physics prize for work on the discontinuous structure of matter, especially discovering sedimentation equilibrium
  Fact 0.35: 1927: Arthur Holly Compton of the United States won the physics prize for discovering the effect named after him
  Fact 0.36: 1927: Charles Thomson Rees Wilson of the United Kingdom shared the physics prize for a method of making charged particle paths visible by condensation of vapour
  Fact 0.37: 1928: Owen Willans Richardson of the United Kingdom won the physics prize for work on the thermionic phenomenon and discovering the law named after him
  Fact 0.38: 1929: Louis Victor Pierre Raymond, 7th Duc de Broglie, of France won the physics prize for discovering the wave nature of electrons
  Fact 0.39: 1930: Chandrasekhara Venkata Raman of India won the physics prize for work on the scattering of light and discovering the effect named after him
  Fact 0.40: 1931: No physics prize was awarded
  Fact 0.41: 1932: Werner Heisenberg of Germany won the physics prize for creating quantum mechanics, whose application led to the discovery of allotropic forms of hydrogen
  Fact 0.42: 1933: Erwin Schrödinger of Austria won the physics prize for discovering new productive forms of atomic theory
  Fact 0.43: 1933: Paul Dirac of the United Kingdom shared the physics prize for work on atomic theory
  Fact 0.44: 1934: No physics prize was awarded
  Fact 0.45: 1935: James Chadwick of the United Kingdom won the physics prize for discovering the neutron
  Fact 0.46: 1936: Victor Francis Hess of Austria won the physics prize for discovering cosmic radiation
  Fact 0.47: 1936: Carl David Anderson of the United States shared the physics prize for discovering the positron
  Fact 0.48: 1937: Clinton Joseph Davisson of the United States shared the physics prize for the experimental discovery of electron diffraction by crystals
  Fact 0.49: 1937: George Paget Thomson of the United Kingdom shared the physics prize for the experimental discovery of electron diffraction by crystals
  Fact 0.50: 1938: Enrico Fermi of Italy won the physics prize for demonstrating new radioactive elements produced by neutron irradiation
  Fact 0.51: 1938: Enrico Fermi won the physics prize for discovering nuclear reactions brought about by slow neutrons
  Fact 0.52: 1939: Ernest Lawrence of the United States won the physics prize for inventing and developing the cyclotron
  Fact 0.53: 1939: Ernest Lawrence won the physics prize for results obtained with the cyclotron, especially on artificial radioactive elements
  Fact 0.54: 1940: No physics prize was awarded because of World War II
  Fact 0.55: 1941: No physics prize was awarded because of World War II
  Fact 0.56: 1942: No physics prize was awarded because of World War II
  Fact 0.57: 1943: Otto Stern of the United States won the physics prize for developing the molecular ray method
  Fact 0.58: 1943: Otto Stern won the physics prize for discovering the magnetic moment of the proton
  Fact 0.59: 1944: Isidor Isaac Rabi, listed as the United States and Poland, won the physics prize for his resonance method for recording the magnetic properties of atomic nuclei
  Fact 0.60: 1945: Wolfgang Pauli of Austria won the physics prize for discovering the Exclusion Principle, also called the Pauli principle
  Fact 0.61: 1946: Percy Williams Bridgman of the United States won the physics prize for inventing an apparatus to produce extremely high pressures
  Fact 0.62: 1946: Percy Williams Bridgman won the physics prize for discoveries he made within high pressure physics
  Fact 0.63: 1947: Edward Victor Appleton of the United Kingdom won the physics prize for investigating the physics of the upper atmosphere
  Fact 0.64: 1947: Edward Victor Appleton won the physics prize especially for discovering the so-called Appleton layer
  Fact 0.65: 1948: Patrick Maynard Stuart Blackett of the United Kingdom won the physics prize for developing the Wilson cloud chamber method
  Fact 0.66: 1948: Patrick Maynard Stuart Blackett won the physics prize for discoveries with that method in nuclear physics and cosmic radiation
  Fact 0.67: 1949: Hideki Yukawa of Japan won the physics prize for predicting the existence of mesons from theoretical work on nuclear forces
  Fact 0.68: 1950: Cecil Frank Powell of the United Kingdom won the physics prize for developing the photographic method of studying nuclear processes
  Fact 0.69: 1950: Cecil Frank Powell won the physics prize for discoveries regarding mesons made with that method
  Fact 0.70: 1951: John Douglas Cockcroft of the United Kingdom shared the physics prize for pioneer work on transmuting atomic nuclei with artificially accelerated particles
  Fact 0.71: 1951: Ernest Thomas Sinton Walton of Ireland shared the physics prize for pioneer work on transmuting atomic nuclei with artificially accelerated particles
  Fact 0.72: 1952: Felix Bloch, listed as Switzerland and the United States, shared the physics prize for developing new methods for nuclear magnetic precision measurements
  Fact 0.73: 1952: Edward Mills Purcell of the United States shared the physics prize for developing new methods for nuclear magnetic precision measurements
  Fact 0.74: 1953: Frits Zernike of the Netherlands won the physics prize for demonstrating the phase contrast method
  Fact 0.75: 1953: Frits Zernike won the physics prize especially for inventing the phase contrast microscope
  Fact 0.76: 1954: Max Born of West Germany won the physics prize for fundamental research in quantum mechanics
  Fact 0.77: 1954: Max Born won the physics prize especially for his statistical interpretation of the wavefunction
  Fact 0.78: 1954: Walther Bothe of West Germany shared the physics prize for the coincidence method and his discoveries made with it
  Fact 0.79: 1955: Willis Eugene Lamb of the United States won the physics prize for discoveries concerning the fine structure of the hydrogen spectrum
  Fact 0.80: 1955: Polykarp Kusch, listed as the United States and Germany, shared the physics prize for precisely determining the magnetic moment of the electron
  Fact 0.81: 1956: John Bardeen of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.82: 1956: Walter Houser Brattain of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.83: 1956: William Bradford Shockley of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.84: 1957: Tsung-Dao Lee of the Republic of China shared the physics prize for investigating the so-called parity laws
  Fact 0.85: 1957: Chen Ning Yang of the Republic of China shared the physics prize for investigating the so-called parity laws
  Fact 0.86: 1958: Pavel Alekseyevich Cherenkov of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.87: 1958: Ilya Frank of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.88: 1958: Igor Yevgenyevich Tamm of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.89: 1959: Emilio Gino Segrè, listed as Italy and the United States, shared the physics prize for discovering the antiproton
  Fact 0.90: 1959: Owen Chamberlain of the United States shared the physics prize for discovering the antiproton
  Fact 0.91: 1960: Donald Arthur Glaser of the United States won the physics prize for inventing the bubble chamber
  Fact 0.92: 1961: Robert Hofstadter of the United States shared the physics prize for pioneering studies of electron scattering in atomic nuclei
  Fact 0.93: 1961: Robert Hofstadter shared the physics prize for discoveries about the structure of the nucleons
  Fact 0.94: 1961: Rudolf Ludwig Mössbauer of West Germany shared the physics prize for researches on the resonance absorption of gamma radiation
  Fact 0.95: 1961: Rudolf Ludwig Mössbauer shared the physics prize for discovering the effect that bears his name
  Fact 0.96: 1962: Lev Davidovich Landau of the Soviet Union won the physics prize for pioneering theories of condensed matter, especially liquid helium
  Fact 0.97: 1963: Eugene Paul Wigner, listed as Hungary and the United States, shared the physics prize for contributions to the theory of the atomic nucleus and elementary particles
  Fact 0.98: 1963: Eugene Paul Wigner shared the physics prize for discovering and applying fundamental symmetry principles
  Fact 0.99: 1963: Maria Goeppert-Mayer of the United States shared the physics prize for discoveries concerning nuclear shell structure
  Fact 0.100: 1963: J. Hans D. Jensen of West Germany shared the physics prize for discoveries concerning nuclear shell structure
  Fact 0.101: 1964: Nicolay Gennadiyevich Basov of the Soviet Union shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.102: 1964: Alexander Prokhorov of the Soviet Union shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.103: 1964: Charles Hard Townes of the United States shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.104: 1965: Richard Phillips Feynman of the United States shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.105: 1965: Julian Schwinger of the United States shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.106: 1965: Shin'ichirō Tomonaga of Japan shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.107: 1966: Alfred Kastler of France won the physics prize for discovering and developing optical methods for studying Hertzian resonances in atoms
  Fact 0.108: 1967: Hans Albrecht Bethe of the United States won the physics prize for contributions to the theory of nuclear reactions
  Fact 0.109: 1967: Hans Albrecht Bethe won the physics prize especially for discoveries about energy production in stars
  Fact 0.110: 1968: Luis Walter Alvarez of the United States won the physics prize for decisive contributions to elementary particle physics
  Fact 0.111: 1968: Luis Walter Alvarez won the physics prize especially for discovering many resonance states using the hydrogen bubble chamber technique
  Fact 0.112: 1969: Murray Gell-Mann of the United States won the physics prize for contributions and discoveries concerning the classification of elementary particles and their interactions
  Fact 0.113: 1970: Hannes Olof Gösta Alfvén of Sweden shared the physics prize for fundamental work in magneto-hydrodynamics
  Fact 0.114: 1970: Louis Néel of France shared the physics prize for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism
  Fact 0.115: 1971: Dennis Gabor, listed as Hungary and the United Kingdom, won the physics prize for inventing and developing the holographic method
  Fact 0.116: 1972: John Bardeen of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.117: 1972: Leon Neil Cooper of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.118: 1972: John Robert Schrieffer of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.119: 1973: Leo Esaki of Japan shared the physics prize for experimental discoveries regarding tunneling phenomena in semiconductors
  Fact 0.120: 1973: Ivar Giaever, listed as the United States and Norway, shared the physics prize for experimental discoveries regarding tunneling phenomena in superconductors
  Fact 0.121: 1973: Brian David Josephson of the United Kingdom shared the physics prize for theoretical predictions of supercurrent properties through a tunnel barrier
  Fact 0.122: 1973: Brian David Josephson shared the physics prize for predicting the phenomena known as the Josephson effect
  Fact 0.123: 1974: Martin Ryle of the United Kingdom shared the physics prize for pioneering research in radio astrophysics
  Fact 0.124: 1974: Martin Ryle shared the physics prize for inventing the aperture synthesis technique
  Fact 0.125: 1974: Antony Hewish of the United Kingdom shared the physics prize for his decisive role in discovering pulsars
  Fact 0.126: 1975: Aage Bohr of Denmark shared the physics prize for discovering the connection between collective motion and particle motion in atomic nuclei
  Fact 0.127: 1975: Ben Roy Mottelson of Denmark shared the physics prize for developing the theory of atomic nucleus structure based on that connection
  Fact 0.128: 1975: Leo James Rainwater of the United States shared the physics prize for developing the theory of atomic nucleus structure based on that connection
  Fact 0.129: 1976: Burton Richter of the United States shared the physics prize for pioneering work in discovering a heavy elementary particle of a new kind
  Fact 0.130: 1976: Samuel Chao Chung Ting of the United States shared the physics prize for pioneering work in discovering a heavy elementary particle of a new kind
  Fact 0.131: 1977: Philip Warren Anderson of the United States shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.132: 1977: Nevill Francis Mott of the United Kingdom shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.133: 1977: John Hasbrouck Van Vleck of the United States shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.134: 1978: Pyotr Leonidovich Kapitsa of the Soviet Union shared the physics prize for basic inventions and discoveries in low-temperature physics
  Fact 0.135: 1978: Arno Allan Penzias of the United States shared the physics prize for discovering cosmic microwave background radiation
  Fact 0.136: 1978: Robert Woodrow Wilson of the United States shared the physics prize for discovering cosmic microwave background radiation
  Fact 0.137: 1979: Sheldon Lee Glashow of the United States shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.138: 1979: Sheldon Lee Glashow shared the physics prize for predicting the weak neutral current
  Fact 0.139: 1979: Abdus Salam of Pakistan shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.140: 1979: Steven Weinberg of the United States shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.141: 1980: James Watson Cronin of the United States shared the physics prize for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons
  Fact 0.142: 1980: Val Logsdon Fitch of the United States shared the physics prize for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons
  Fact 0.143: 1981: Nicolaas Bloembergen, listed as the Netherlands and the United States, shared the physics prize for contributions to laser spectroscopy
  Fact 0.144: 1981: Arthur Leonard Schawlow of the United States shared the physics prize for contributions to laser spectroscopy
  Fact 0.145: 1981: Kai Manne Börje Siegbahn of Sweden shared the physics prize for contributions to high-resolution electron spectroscopy
  Fact 0.146: 1982: Kenneth G. Wilson of the United States won the physics prize for his theory of critical phenomena in connection with phase transitions
  Fact 0.147: 1982: Subrahmanyan Chandrasekhar, listed as India and the United States, shared the physics prize for theoretical studies of physical processes important to the structure and evolution of stars
  Fact 0.148: 1982: William Alfred Fowler of the United States shared the physics prize for theoretical and experimental studies of nuclear reactions important in forming the chemical elements
  Fact 0.149: 1984: Carlo Rubbia of Italy shared the physics prize for decisive contributions to the large project that discovered the W and Z field particles
  Fact 0.150: 1984: Simon van der Meer of the Netherlands shared the physics prize for decisive contributions to the large project that discovered the W and Z field particles
  Fact 0.151: 1985: Klaus von Klitzing of West Germany won the physics prize for discovering the quantized Hall effect
  Fact 0.152: 1986: Ernst Ruska of West Germany shared the physics prize for fundamental work in electron optics
  Fact 0.153: 1986: Ernst Ruska shared the physics prize for designing the first electron microscope
  Fact 0.154: 1986: Gerd Binnig of West Germany shared the physics prize for designing the scanning tunneling microscope
  Fact 0.155: 1986: Heinrich Rohrer of Switzerland shared the physics prize for designing the scanning tunneling microscope
  Fact 0.156: 1986: Johannes Georg Bednorz of West Germany shared the physics prize for a breakthrough in discovering superconductivity in ceramic materials
  Fact 0.157: 1986: Karl Alexander Müller of Switzerland shared the physics prize for a breakthrough in discovering superconductivity in ceramic materials
  Fact 0.158: 1988: Leon Max Lederman of the United States shared the physics prize for the neutrino beam method
  Fact 0.159: 1988: Leon Max Lederman shared the physics prize for demonstrating the doublet structure of the leptons through discovering the muon neutrino
  Fact 0.160: 1988: Melvin Schwartz of the United States shared the physics prize for the neutrino beam method and the muon neutrino discovery
  Fact 0.161: 1988: Jack Steinberger of the United States shared the physics prize for the neutrino beam method and the muon neutrino discovery
  Fact 0.162: 1989: Norman Foster Ramsey of the United States shared the physics prize for inventing the separated oscillatory fields method
  Fact 0.163: 1989: Norman Foster Ramsey shared the physics prize for its use in the hydrogen maser and other atomic clocks
  Fact 0.164: 1989: Hans Georg Dehmelt, listed as the United States and Germany, shared the physics prize for developing the ion trap technique
  Fact 0.165: 1989: Wolfgang Paul of West Germany shared the physics prize for developing the ion trap technique
  Fact 0.166: 1990: Jerome I. Friedman of the United States shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.167: 1990: Henry Way Kendall of the United States shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.168: 1990: Richard E. Taylor of Canada shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.169: 1991: Pierre-Gilles de Gennes of France won the physics prize for discovering that methods for order phenomena in simple systems generalize to liquid crystals and polymers
  Fact 0.170: 1992: Georges Charpak, listed as France and Poland, won the physics prize for inventing and developing particle detectors
  Fact 0.171: 1992: Georges Charpak won the physics prize especially for the multiwire proportional chamber
  Fact 0.172: 1993: Russell Alan Hulse of the United States shared the physics prize for discovering a new type of pulsar
  Fact 0.173: 1993: Joseph Hooton Taylor Jr. of the United States shared the physics prize for discovering a new type of pulsar
  Fact 0.174: 1994: Bertram Brockhouse of Canada shared the physics prize for developing neutron spectroscopy
  Fact 0.175: 1994: Bertram Brockhouse shared the physics prize for pioneering contributions to neutron scattering techniques for studies of condensed matter
  Fact 0.176: 1994: Clifford Glenwood Shull of the United States shared the physics prize for developing the neutron diffraction technique
  Fact 0.177: 1994: Clifford Glenwood Shull shared the physics prize for pioneering contributions to neutron scattering techniques for studies of condensed matter
  Fact 0.178: 1995: Martin Lewis Perl of the United States shared the physics prize for discovering the tau lepton
  Fact 0.179: 1995: Martin Lewis Perl shared the physics prize for pioneering experimental contributions to lepton physics
  Fact 0.180: 1995: Frederick Reines of the United States shared the physics prize for detecting the neutrino
  Fact 0.181: 1995: Frederick Reines shared the physics prize for pioneering experimental contributions to lepton physics
  Fact 0.182: 1995: David Morris Lee of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.183: 1995: Douglas D. Osheroff of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.184: 1995: Robert Coleman Richardson of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.185: 1997: Steven Chu of the United States shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.186: 1997: Claude Cohen-Tannoudji of France shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.187: 1997: William Daniel Phillips of the United States shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.188: 1998: Robert B. Laughlin of the United States shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.189: 1998: Horst Ludwig Störmer of Germany shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.190: 1998: Daniel Chee Tsui, listed as the Republic of China and the United States, shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.191: 1999: Gerard 't Hooft of the Netherlands shared the physics prize for elucidating the quantum structure of electroweak interactions
  Fact 0.192: 1999: Martinus J.G. Veltman of the Netherlands shared the physics prize for elucidating the quantum structure of electroweak interactions
  Fact 0.193: 2000: Zhores Ivanovich Alferov of Russia shared the physics prize for developing semiconductor heterostructures used in high-speed and optoelectronics
  Fact 0.194: 2000: Herbert Kroemer of Germany shared the physics prize for developing semiconductor heterostructures used in high-speed and optoelectronics
  Fact 0.195: 2000: Jack St. Clair Kilby of the United States shared the physics prize for his part in the invention of the integrated circuit
  Fact 0.196: 2001: Eric Allin Cornell of the United States shared the physics prize for achieving Bose-Einstein condensation in dilute gases of alkali atoms
  Fact 0.197: 2001: Carl Edwin Wieman of the United States shared the physics prize for early fundamental studies of the properties of the condensates
  Fact 0.198: 2001: Wolfgang Ketterle of Germany shared the physics prize for achieving Bose-Einstein condensation in dilute gases of alkali atoms
  Fact 0.199: 2002: Raymond Davis Jr. of the United States shared the physics prize for pioneering contributions to astrophysics through the detection of cosmic neutrinos
  Fact 0.200: 2002: Masatoshi Koshiba of Japan shared the physics prize for pioneering contributions to astrophysics through the detection of cosmic neutrinos
  Fact 0.201: 2002: Riccardo Giacconi, listed as Italy and the United States, shared the physics prize for pioneering contributions to astrophysics that led to the discovery of cosmic X-ray sources
  Fact 0.202: 2003: Alexei Alexeyevich Abrikosov, listed as Russia and the United States, shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.203: 2003: Vitaly Lazarevich Ginzburg of Russia shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.204: 2003: Anthony James Leggett, listed as the United Kingdom and the United States, shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.205: 2003: David J. Gross of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.206: 2003: Hugh David Politzer of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.207: 2003: Frank Wilczek of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.208: 2005: Roy J. Glauber of the United States won the physics prize for his contribution to the quantum theory of optical coherence
  Fact 0.209: 2005: John L. Hall of the United States shared the physics prize for contributions to laser-based precision spectroscopy, including the optical frequency comb technique
  Fact 0.210: 2005: Theodor W. Hänsch of Germany shared the physics prize for contributions to laser-based precision spectroscopy, including the optical frequency comb technique
  Fact 0.211: 2006: John C. Mather of the United States shared the physics prize for discovering the blackbody form of the cosmic microwave background radiation
  Fact 0.212: 2006: John C. Mather shared the physics prize for discovering the anisotropy of the cosmic microwave background radiation
  Fact 0.213: 2006: George F. Smoot of the United States shared the physics prize for discovering the blackbody form of the cosmic microwave background radiation
  Fact 0.214: 2006: George F. Smoot shared the physics prize for discovering the anisotropy of the cosmic microwave background radiation
  Fact 0.215: 2007: Albert Fert of France shared the physics prize for discovering giant magnetoresistance
  Fact 0.216: 2007: Peter Grünberg of Germany shared the physics prize for discovering giant magnetoresistance
  Fact 0.217: 2008: Makoto Kobayashi of Japan shared the physics prize for discovering the origin of the broken symmetry that predicts at least three families of quarks in nature
  Fact 0.218: 2008: Toshihide Maskawa of Japan shared the physics prize for discovering the origin of the broken symmetry that predicts at least three families of quarks in nature
  Fact 0.219: 2008: Yoichiro Nambu, listed as Japan and the United States, shared the physics prize for discovering the mechanism of spontaneous broken symmetry in subatomic physics
  Fact 0.220: 2009: Charles K. Kao, listed as Hong Kong, the United Kingdom and the United States, shared the physics prize for groundbreaking achievements concerning light transmission in fibers for optical communication
  Fact 0.221: 2009: Willard S. Boyle, listed as Canada and the United States, shared the physics prize for inventing an imaging semiconductor circuit, the CCD sensor
  Fact 0.222: 2009: George E. Smith of the United States shared the physics prize for inventing an imaging semiconductor circuit, the CCD sensor
  Fact 0.223: 2010: Andre Geim, listed as Russia, the United Kingdom and the Netherlands, shared the physics prize for groundbreaking experiments regarding the two-dimensional material graphene
  Fact 0.224: 2010: Konstantin Novoselov, listed as Russia and the United Kingdom, shared the physics prize for groundbreaking experiments regarding the two-dimensional material graphene
  Fact 0.225: 2011: Saul Perlmutter of the United States shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.226: 2011: Brian P. Schmidt, listed as Australia and the United States, shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.227: 2011: Adam G. Riess of the United States shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.228: 2012: Serge Haroche of France shared the physics prize for ground-breaking experimental methods that enable measuring and manipulating individual quantum systems
  Fact 0.229: 2012: David J. Wineland of the United States shared the physics prize for ground-breaking experimental methods that enable measuring and manipulating individual quantum systems
  Fact 0.230: 2013: François Englert of Belgium shared the physics prize for the theoretical discovery of a mechanism explaining the origin of mass of subatomic particles
  Fact 0.231: 2013: Peter Higgs of the United Kingdom shared the physics prize for the theoretical discovery of a mechanism explaining the origin of mass of subatomic particles
  Fact 0.232: 2014: Isamu Akasaki of Japan shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.233: 2014: Hiroshi Amano of Japan shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.234: 2014: Shuji Nakamura, listed as Japan and the United States, shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.235: 2015: Takaaki Kajita of Japan shared the physics prize for discovering neutrino oscillations, which shows that neutrinos have mass
  Fact 0.236: 2015: Arthur B. McDonald of Canada shared the physics prize for discovering neutrino oscillations, which shows that neutrinos have mass
  Fact 0.237: 2016: David J. Thouless, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.238: 2016: F. Duncan M. Haldane, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.239: 2016: John M. Kosterlitz, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.240: 2017: Rainer Weiss, listed as Germany and the United States, shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
  Fact 0.241: 2017: Kip Thorne of the United States shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
  Fact 0.242: 2017: Barry Barish of the United States shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
Sentence 1: The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) .
  Fact 1.0: The first Nobel Prize in Physics was awarded in 1901
  Fact 1.1: The first Nobel Prize in Physics went to Wilhelm Conrad Röntgen of Germany
  Fact 1.2: Wilhelm Conrad Röntgen received 150,782 SEK
  Fact 1.3: 150,782 SEK is equal to 7,731,004 SEK in December 2007
  Fact 1.4: John Bardeen is the only laureate to win the prize twice
  Fact 1.5: John Bardeen won the prize in 1956 and in 1972
  Fact 1.6: Maria Skłodowska-Curie also won two Nobel Prizes
  Fact 1.7: Maria Skłodowska-Curie won the physics prize in 1903
  Fact 1.8: Maria Skłodowska-Curie won the chemistry prize in 1911
  Fact 1.9: William Lawrence Bragg was, until October 2014, the youngest ever Nobel laureate
  Fact 1.10: William Lawrence Bragg won the prize in 1915 at the age of 25
  Fact 1.11: Two women have won the prize: Curie and Maria Goeppert-Mayer
  Fact 1.12: Maria Goeppert-Mayer won the prize in 1963
  Fact 1.13: As of 2017, the prize has been awarded to 206 individuals
  Fact 1.14: There have been six years in which the Nobel Prize in Physics was not awarded: 1916, 1931, 1934, 1940, 1941 and 1942

Retrieved items:
<item 1>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1910 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1911 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1912 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1913 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1914 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1915 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1917 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1918 '' .
</item 1>

<item 2>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1925 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1926 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1927 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1928 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1929 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1930 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1932 ''
. Nobel Foundation . Retrieved 2008 - 10 - 09 .
</item 2>

<item 3>
List of Nobel laureates in Physics
Thomson 1907 Michelson 1908 Lippmann 1909 Marconi / Braun 1910 Van der Waals 1911 Wien 1912 Dalén 1913 Kamerlingh Onnes 1914 Laue 1915 W.L. Bragg / W.H. Bragg
1916 1917 Barkla 1918 Planck 1919 Stark 1920 Guillaume 1921 Einstein 1922 N. Bohr 1923 Millikan 1924 M. Siegbahn 1925 Franck / Hertz 1926 -- 1950 1926 Perrin
1927 Compton / C. Wilson 1928 O. Richardson 1929 De Broglie 1930 Raman 1931 1932 Heisenberg 1933 Schrödinger / Dirac 1934 1935 Chadwick 1936 Hess / C.D.
Anderson 1937 Davisson / G.P.
</item 3>

<item 4>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1918 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1919 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1920 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1921 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1922 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1923 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1924 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1925 '' .
</item 4>

<item 5>
List of Nobel laureates in Physics
Archived from the original on 2008 - 08 - 22 . Retrieved 2008 - 10 - 29 . Jump up ^ `` The Nobel Prize '' . Nobel Foundation . Retrieved 2008 - 10 - 29 . Jump
up ^ `` Nobel Laureates Facts '' . Nobel Foundation . Archived from the original on 2007 - 02 - 02 . Retrieved 2008 - 10 - 29 . Jump up ^ `` Women Nobel
Laureates '' . Nobel Foundation . Retrieved 2008 - 10 - 29 . Jump up ^ `` The Nobel Prize in Physics 1901 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 .
Jump up ^ `` The Nobel Prize in Physics 1902 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1903 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 .
</item 5>

<item 6>
List of Nobel laureates in Physics
Contents ( hide ) 1 Statistics 2 Laureates 3 See also 4 References 4.1 Notes 4.2 Citations 4.3 Sources 5 External links Statistics ( edit ) The first Nobel
Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John
Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry
in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won
the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals .
</item 6>

<item 7>
List of Nobel laureates in Physics
As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934
, 1940 -- 1942 ) . Laureates ( edit ) Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has
rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service
they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France ``
for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri
Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and
for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays ''
1906 Joseph
</item 7>

<item 8>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 2016 '' . Nobel Foundation . Retrieved 2016 - 10 - 04 . Jump up ^ `` The Nobel Prize in Physics 2017 '' . Nobel
Foundation . Retrieved 2017 - 10 - 03 . Sources ( edit ) `` All Nobel Laureates in Physics '' . Nobel Foundation . Retrieved 2008 - 10 - 08 . External links (
edit ) Wikimedia Commons has media related to Nobel laureates in Physics . Official website of the Royal Swedish Academy of Sciences Official website of the
Nobel Foundation Laureates of the Nobel Prize in Physics 1901 -- 1925 1901 Röntgen 1902 Lorentz / Zeeman 1903 Becquerel / P. Curie / M. Curie 1904 Rayleigh 1905
Lenard 1906 J.J.
</item 8>

<item 9>
List of Nobel laureates in Physics
von Lenard Austria - Hungary Germany `` for his work on cathode rays '' 1906 Joseph John Thomson United Kingdom `` for his theoretical and experimental
investigations on the conduction of electricity by gases '' 1907 Albert Abraham Michelson United States Poland `` for his optical precision instruments and the
spectroscopic and metrological investigations carried out with their aid '' 1908 Gabriel Lippmann France `` for his method of reproducing colours
photographically based on the phenomenon of interference '' 1909 Guglielmo Marconi Italy `` for their contributions to the development of wireless telegraphy ''
Karl Ferdinand Braun Germany 1910 Johannes Diderik van der Waals Netherlands `` for his work on the equation of state for gases and liquids '' 1911 Wilhelm Wien
Germany `` for his discoveries regarding the laws governing the radiation of heat '' 1912 Nils Gustaf Dalén Sweden `` for his invention of automatic valves
designed to be used in combination with gas accumulators in lighthouses and buoys '' 1913 Heike Kamerlingh - Onnes Netherlands `` for his investigations on the
properties of matter at low temperatures which led , inter alia ,
</item 9>

<item 10>
List of Nobel laureates in Physics
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1904 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1905 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1906 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1907 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1908 ''
. Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1909 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1910 '' . Nobel Foundation .
</item 10>
=== END ITEM id=9d446d266e419f87 ===

=== ITEM id=89a8bfaef2936d10 ===
Gold sentences and facts:
Sentence 0: Only Mexico , Italy , France , Germany ( West Germany until shortly after the 1990 World Cup ) and Brazil have hosted the event on two occasions . Mexico City 's Estadio Azteca and Rio de Janeiro 's Maracanã are the only venues ever to have hosted two FIFA World Cup finals . Only the 2002 FIFA World Cup had more than one host , being split between Japan and South Korea .
  Fact 0.0: Only Mexico has hosted the event on two occasions.
  Fact 0.1: Only Italy has hosted the event on two occasions.
  Fact 0.2: Only France has hosted the event on two occasions.
  Fact 0.3: Only Germany has hosted the event on two occasions.
  Fact 0.4: Only Brazil has hosted the event on two occasions.
  Fact 0.5: Germany's hosting covers West Germany until shortly after the 1990 World Cup.
  Fact 0.6: Mexico City's Estadio Azteca is one of the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.7: Rio de Janeiro's Maracanã is one of the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.8: Estadio Azteca and the Maracanã are the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.9: Only the 2002 FIFA World Cup had more than one host.
  Fact 0.10: The 2002 FIFA World Cup was split between Japan and South Korea.

Retrieved items:
<item 1>
FIFA World Cup hosts
( PDF ) . PricewaterhouseCoopers . May 2010 . pp. 6 -- 7 . Retrieved 2 August 2014 . Jump up ^ Smallwood , John ( 11 June 2014 ) . `` Ultimate home field
advantage : Host nation luck '' . Philadelphia Daily News . Retrieved 2 August 2014 . External links ( edit ) FIFA World Cup Host Announcement Decision ( PDF )
FIFA World Cup hosts 1930 : Uruguay 1934 : Italy 1938 : France 1950 : Brazil 1954 : Switzerland 1958 : Sweden 1962 : Chile 1966 : England 1970 : Mexico 1974 :
West Germany 1978 : Argentina 1982 : Spain 1986 : Mexico 1990 : Italy 1994 : United States 1998 : France 2002 : South Korea & Japan 2006 : Germany 2010 : South
Africa 2014 : Brazil 2018 : Russia 2022 : Qatar FIFA World Cup Tournaments Uruguay 1930 Italy 1934 France 1938 Brazil 1950 Switzerland 1954 Sweden 1958 Chile
1962 England
</item 1>

<item 2>
FIFA World Cup hosts
33 1.3 12 Fourth place South Korea 2002 7 57 % 8 6 0.3 1.1 13 Fourth place Brazil 2014 7 57 % 11 14 − 3 − 0.4 1.57 14 Quarter - final Mexico 1986 5 0 80 % 6 0.8
15 Quarter - final Mexico 1970 63 % 6 0.5 1.5 16 Quarter - final Switzerland 1954 0 50 % 11 11 0 0 2.8 17 Quarter - final France 1938 0 50 % 0 0 18 Second round
Spain 1982 5 40 % 5 − 1 − 0.2 0.8 19 Round of 16 Japan 2002 63 % 5 0.5 1.25 20 Round of 16 United States 1994 38 % − 1 − 0.3 0.8 21 Group stage South Africa 50
% 5 − 2 − 0.
</item 2>

<item 3>
FIFA World Cup hosts
Contents ( hide ) 1 List of hosts 2 1930 FIFA World Cup 3 1934 FIFA World Cup 4 1938 FIFA World Cup 5 Cancelled FIFA World Cups 1942 and 1946 6 1950 and 1954
FIFA World Cups 6.1 1949 bid 6.2 1951 bid 7 1958 FIFA World Cup 8 1962 FIFA World Cup 9 1966 FIFA World Cup 10 1970 FIFA World Cup 11 1974 , 1978 , 1982 FIFA
World Cups 11.1 1974 results 11.2 1978 results 11.3 1982 results 12 1986 FIFA World Cup 13 1990 FIFA World Cup 14 1994 FIFA World Cup 15 1998 FIFA World Cup 16
2002 FIFA World Cup 17 2006 FIFA World Cup 17.1 Controversy 18 2010 FIFA World Cup 18.1 Controversy 19 2014 FIFA World Cup 20 2018 and 2022 FIFA World Cups 20.1
Accusations of bribery 21
</item 3>

<item 4>
FIFA World Cup hosts
Both Argentina and Uruguay thus boycotted the tournament . The first tournament following World War II , held in Brazil in 1950 , had three teams withdraw due
to either financial problems or disagreements with the organization . To avoid any future boycotts or controversy , FIFA began a pattern of alternation between
the Americas and Europe , which continued until the 2002 FIFA World Cup in Asia . The system evolved so that the host country is now chosen in a vote by FIFA 's
Council ( former Executive Committee ) . This is done under an exhaustive ballot system . The decision is currently made roughly seven years in advance of the
tournament , though the hosts for the 2022 tournament were chosen at the same time as those for the 2018 tournament . Only Mexico , Italy , France , Germany (
West Germany until shortly after the 1990 World Cup ) and Brazil have hosted the event on two occasions . Mexico City 's Estadio Azteca and Rio de Janeiro 's
Maracanã are the only venues ever to have hosted two FIFA World Cup finals . Only the 2002 FIFA World Cup had more than one host , being split between Japan and
South Korea .
</item 4>

<item 5>
FIFA World Cup hosts
^ Jump up to : Joint bid by Libya and Tunisia . Host country performances ( edit ) Further information : National team appearances in the FIFA World Cup It is
widely considered that home advantage is common in the World Cup , with the host team usually performing above average . Of the eight countries that have won
the tournament , only Brazil and Spain were not champions at home , and both England and France had their only titles as hosts . Further , Sweden got to its
only final as hosts of the 1958 tournament , and only the Swedes and Brazilians finished as runner - up in home soil . The host country reached the semi-finals
thirteen times in twenty tournaments , and both Chile and South Korea had their only finishes on the top four at home . Only South Africa in 2010 managed to not
go past the group stage . # Performance Team Year Pld Win % GF GA GD GD / M GF / M Champion Uruguay 1930 0 0 100 % 15 12 3.8 Champion France 1998 7 6 0 93 % 15
13 1.9 2.
</item 5>

<item 6>
FIFA World Cup hosts
1962 , 1966 , 1974 , 1982 , 2006 Argentina 5 1938 , 1962 , 1970 , 1978 , 2030 Mexico 1970 , 1978 , 1986 , 2002 , 2026 Morocco 1994 , 1998 , 2006 , 2010 , 2026
Spain 1930 , 1966 , 1974 , 1982 , 2018 Brazil 1950 , 1994 , 2006 , 2014 England 1966 , 1990 , 2006 , 2018 United States 1986 , 1994 , 2022 , 2026 Italy 1930 ,
1934 , 1990 Japan 1970 , 2002 , 2022 Canada 1986 , 2026 Chile 1962 , 2014 Colombia 1986 , 2014 France 1938 , 1998 Netherlands 1930 , 2018 Russia 1990 , 2018
South Africa 2006 , 2010 South Korea 2002 , 2022 Switzerland 1954 ,
</item 6>

<item 7>
FIFA World Cup hosts
FIFA World Cup hosts - wikipedia FIFA World Cup hosts Jump to : navigation , search Map of FIFA World Cup final hosts , 1930 -- 2014 . Dark green : twice ;
middle green : once ; light green : planned until 2022 Sixteen countries have been FIFA World Cup hosts in the competition 's twenty tournaments since the
inaugural World Cup in 1930 . The organization at first awarded hosting to countries at meetings of FIFA 's congress . The choice of location was controversial
in the earliest tournaments , given the three - week boat journey between South America and Europe , the two centers of strength in football at the time . The
decision to hold the first cup in Uruguay , for example , led to only four European nations competing . The next two World Cups were both held in Europe . The
decision to hold the second of these , the 1938 FIFA World Cup , in France was controversial , as the South American countries had been led to understand that
the World Cup would rotate between the two continents . Both Argentina and Uruguay thus boycotted the tournament .
</item 7>

<item 8>
FIFA World Cup hosts
FIFA World Cup bids 1930 Hungary Italy Netherlands Spain Sweden Uruguay 1934 Italy Sweden 1938 Argentina France Germany 1942 Brazil Germany 1946 None 1950
Brazil 1954 Switzerland 1958 Sweden 1962 Argentina Chile West Germany 1966 England West Germany Spain 1970 Argentina Mexico Italy Netherlands Spain West Germany
1978 Argentina Mexico 1982 Spain West Germany 1986 Canada Colombia Mexico United States 1990 Austria England France Greece Italy Soviet Union West Germany
Yugoslavia 1994 Brazil Morocco United States 1998 France Morocco Switzerland 2002 Japan -- South Korea Mexico 2006 Brazil England Germany Morocco South Africa
Egypt Libya -- Tunisia Morocco Nigeria South Africa 2014 Brazil Colombia 2018 Belgium -- Netherlands England Portugal -- Spain Russia 2022 Australia Japan Qatar
South Korea United States 2026 Canada -- Mexico -- United States Morocco 2030 Argentina -- Paraguay -- Uruguay England Notes : Bold entries are those chosen as
hosts Event cancelled due to World War II Colombia withdrew after being award event , which was re-awarded to Mexico Retrieved from ``
https://en.wikipedia.org/w/index.
</item 8>

<item 9>
FIFA World Cup hosts
1 Accusations of bribery 21 2026 FIFA World Cup 22 2030 FIFA World Cup 23 2034 FIFA World Cup 24 Total bids by country 25 Host country performances 26
References 27 External links List of hosts ( edit ) Year Host Continent Winner 1930 Uruguay South America Uruguay 1934 Italy Europe Italy 1938 France Europe
Italy 1942 Cancelled due to World War II 1946 Cancelled due to World War II 1950 Brazil South America Uruguay 1954 Switzerland Europe West Germany 1958 Sweden
Europe Brazil 1962 Chile South America Brazil 1966 England Europe England 1970 Mexico North America Brazil West Germany Europe West Germany 1978 Argentina South
America Argentina 1982 Spain Europe Italy 1986 Mexico North America Argentina 1990 Italy Europe West Germany 1994 United States North America Brazil 1998 France
Europe France 2002 South Korea / Japan Asia Brazil 2006 Germany Europe Italy South Africa Africa Spain 2014 Brazil South America Germany 2018 Russia Europe 2022
Qatar Asia 2026 TBD 2030 TBD 1930 FIFA World Cup ( edit ) Main article : 1930 FIFA World Cup Bids : Hungary Italy
</item 9>

<item 10>
FIFA World Cup hosts
1938 Brazil 1950 Switzerland 1954 Sweden 1958 Chile 1962 England 1966 Mexico 1970 West Germany 1974 Argentina 1978 Spain 1982 Mexico 1986 Italy 1990 United
States 1994 France 1998 South Korea / Japan 2002 Germany 2006 South Africa 2010 Brazil 2014 Russia 2018 Qatar 2022 2026 2030 2034 Qualification 1930 1934 1938
1950 1954 1958 1962 1966 1970 1978 1982 1986 1990 1994 1998 2002 2006 2014 2018 Finals 1930 1934 1938 1950 1954 1958 1962 1966 1970 1978 1982 1986 1990 1994
1998 2002 2006 2014 2018 Squads 1930 1934 1938 1950 1954 1958 1962 1966 1970 1978 1982 1986
</item 10>
=== END ITEM id=89a8bfaef2936d10 ===

=== ITEM id=44a589e7b0d3612c ===
Gold sentences and facts:
Sentence 0: Year Winning team Losing team Score Site Notes 1909 Pittsburgh Pirates Detroit Tigers 8 -- 0 Bennett Park , Detroit Rookie Babe Adams tosses a six hit shutout for his third win of the series . 1912 Boston Red Sox New York Giants 3 -- 2 F / 10 Fenway Park , Boston This decisive contest was actually game eight , as game two had ended in a tie . 1924 Washington Senators New York Giants 4 -- 3 F / 12 Griffith Stadium , Washington , D.C. Longest game seven in innings . In the 12th , Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy . Ruel scored on a game - winning RBI double by Earl McNeely . 1925 Pittsburgh Pirates Washington Senators 9 -- 7 Forbes Field , Pittsburgh Pittsburgh became the first team in a best of seven to win the series when trailing three games to one . 1926 St. Louis Cardinals New York Yankees 3 -- 2 Yankee Stadium , the Bronx , New York City Grover Cleveland Alexander strikes out Tony Lazzeri with the bases loaded to end the seventh and preserve the Cardinals lead . Two innings later Babe Ruth is caught stealing to end the World Series . 1931 St. Louis Cardinals Philadelphia Athletics 4 -- 2 St. Louis , Missouri Last postseason game for the Athletics while based in Philadelphia . 1934 St. Louis Cardinals Detroit Tigers 11 -- 0 Navin Field , Detroit Dizzy Dean 's second win , a six - hit shutout on one day of rest , gives the Cardinals the title . 1940 Cincinnati Reds Detroit Tigers 2 -- 1 Crosley Field , Cincinnati This game was over in a snappy 1 : 47 while a sacrifice bunt and sacrifice fly scored the winning runs . 1945 Detroit Tigers Chicago Cubs 9 -- 3 Wrigley Field , Chicago Cubs last World Series appearance until 2016 . See also : Curse of the Billy Goat 1946 St. Louis Cardinals Boston Red Sox 4 -- 3 Sportsman 's Park III , St. Louis , Missouri Enos Slaughter 's famous mad dash scored the deciding run to win the game and the series . 1947 New York Yankees Brooklyn Dodgers 5 -- 2 Yankee Stadium , the Bronx , New York City This was the Series that featured Cookie Lavagetto 's pinch - hit that spoiled Bill Bevens ' no - hit bid and Al Gionfriddo 's catch to rob Joe DiMaggio of a home run . 1952 New York Yankees Brooklyn Dodgers 4 -- 2 Ebbets Field , Brooklyn , New York City Billy Martin 's game saving infield catch preserves the Yankees lead and helps win the game and the series . 1955 Brooklyn Dodgers New York Yankees 2 -- 0 Yankee Stadium , the Bronx , New York City Dodgers ' only championship in Brooklyn . The home team won all games of the series , except for game seven . 1956 New York Yankees Brooklyn Dodgers 9 -- 0 Ebbets Field , Brooklyn , New York City Jackie Robinson 's final Major League game . Last postseason game at Ebbets Field and Dodgers ' last postseason game before moving to Los Angeles . The home team won all games of the series , except for game seven . 1957 Milwaukee Braves New York Yankees 5 -- 0 Yankee Stadium , the Bronx , New York City The Braves ' only championship in Milwaukee . 1958 New York Yankees Milwaukee Braves 6 -- 2 County Stadium , Milwaukee The Yankees win their 18th World Series title after being down three games to one . This was the Braves ' last postseason game in Milwaukee . 1960 Pittsburgh Pirates New York Yankees 10 -- 9 Forbes Field , Pittsburgh Bill Mazeroski hit the walk - off home run to win the Pirates the championship . Last postseason game in Forbes Field . Last postseason game for Yankees ' manager Casey Stengel . Highest scoring game seven . 1962 New York Yankees San Francisco Giants 1 -- 0 Candlestick Park , San Francisco With runners on second and third and two outs , Willie McCovey 's line drive to Bobby Richardson ended the game . First game seven that ended with a 1 - 0 score . 1964 St. Louis Cardinals New York Yankees 7 -- 5 Busch Stadium I , St. Louis , Missouri The Yankees ' last World Series appearance until 1976 . Last postseason game at Busch Stadium I . 1965 Los Angeles Dodgers Minnesota Twins 2 -- 0 Metropolitan Stadium , Bloomington , Minnesota Pitching through immense pain and on two days rest , Sandy Koufax throws a 3 hit shutout to win the Series and World Series MVP . The home team won all games of the series , except for game seven . 1967 St. Louis Cardinals Boston Red Sox 7 -- 2 Fenway Park , Boston Bob Gibson wins his third game of the Series and also homers . 1968 Detroit Tigers St. Louis Cardinals 4 -- 1 Busch Stadium II , St. Louis , Missouri Mickey Lolich out - duels Bob Gibson . The Tigers rallied from a 3 - 1 deficit to win the series . First game seven loss by the Cardinals in the World Series . 1971 Pittsburgh Pirates Baltimore Orioles 2 -- 1 Memorial Stadium , Baltimore The home team won all games of the series , except for game seven . 1972 Oakland Athletics Cincinnati Reds 3 -- 2 Riverfront Stadium , Cincinnati The A 's first championship in Oakland . The A 's were the last American League team to win a World Series game seven on the road until 2017 . 1973 Oakland Athletics New York Mets 5 -- 2 Oakland -- Alameda County Coliseum , Oakland , California Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game to set the tone for victory and the series . Cincinnati Reds Boston Red Sox 4 -- 3 Fenway Park , Boston One night after Carlton Fisk 's famous home run , the Reds rebound to take the Series . 1979 Pittsburgh Pirates Baltimore Orioles 4 -- 1 Memorial Stadium , Baltimore The Pittsburgh Pirates , powered by NLCS and World Series MVP Willie Stargell , rally from a 3 - 1 deficit to win the series . The Pirates were the last team to win a road game seven until 2014 . 1982 St. Louis Cardinals Milwaukee Brewers 6 -- 3 Busch Stadium II , St. Louis , Missouri Joaquín Andújar earns his second win of the Series and Bruce Sutter his second save . 1985 Kansas City Royals St. Louis Cardinals 11 -- 0 Royals Stadium , Kansas City , Missouri Second World Series held entirely in the state of Missouri . The Royals also won the 1985 American League Championship Series in seven games . Royals became first team to win the World Series after losing the first two games at home . The Royals were the last team to come back from 3 -- 1 down to win a title , in any of the five major men 's professional sports leagues , until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3 -- 1 . The 1985 Royals were the last team to win a World Series after trailing 3 -- 1 until the Cubs did so in 2016 . 1986 New York Mets Boston Red Sox 8 -- 5 Shea Stadium , Flushing , Queens , New York Mets become the first team to be one strike away from elimination and come back to win the series . They also become the second team to win the series after losing the first two at home . Minnesota Twins St. Louis Cardinals 4 -- 2 Hubert H. Humphrey Metrodome , Minneapolis The home team won all seven games of the series for the first time in MLB history . 1991 Minnesota Twins Atlanta Braves 1 -- 0 F / 10 Hubert H. Humphrey Metrodome , Minneapolis Like in 1987 , the home team won all games of the series . The Twins won in the bottom of the 10th on a walk - off single by Gene Larkin scoring Dan Gladden after ​ 9 ⁄ scoreless innings of play . Jack Morris goes all ten innings for the Twins and is named the Series MVP . Twins Centerfielder Kirby Puckett hit a famous walk - off home run in the bottom of the eleventh inning of the previous night 's game to force this decisive game . 1997 Florida Marlins Cleveland Indians 3 -- 2 F / 11 Pro Player Stadium , Miami Gardens , Florida Florida becomes the first wild card team to win the World Series . The Marlins rallied for one run in the bottom of the ninth to tie and won in the eleventh on an Édgar Rentería RBI single off Charles Nagy . 2001 Arizona Diamondbacks New York Yankees 3 -- 2 Bank One Ballpark , Phoenix , Arizona Arizona becomes the fastest expansion team to win the World Series , rallying for two runs in the bottom of the ninth for a come - from - behind win , capped off by Luis Gonzalez hitting the winning RBI single off closer Mariano Rivera . The home team won every game of the series . 2002 Anaheim Angels San Francisco Giants 4 -- 1 Edison International Field , Anaheim , California Both World Series teams were wild card entrants into the playoffs . John Lackey became the second rookie pitcher to win a World Series game seven . 2011 St. Louis Cardinals Texas Rangers 6 -- 2 Busch Stadium , St. Louis , Missouri Cardinals became the second team to be one strike away from elimination and come back to win the series , which they did twice . Just like the 1991 series , a walk - off home run in the bottom of the eleventh inning of game six , forced this decisive game with David Freese providing the fire power . Most recent game seven won by the home team . 2014 San Francisco Giants Kansas City Royals 3 -- 2 Kauffman Stadium , Kansas City , Missouri Like in 2002 , both World Series teams were wild card entrants into the playoffs . The Giants became the first NL team to win a game seven on the road since 1979 . NLCS MVP and eventual World Series MVP Madison Bumgarner , normally a starter , pitched five scoreless innings of relief on two days rest to save the Giants ' third title in five years . 2016 Chicago Cubs Cleveland Indians 8 -- 7 F / 10 Progressive Field , Cleveland , Ohio Chicago Cubs ' first World Series win since 1908 . Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City Royals in 1985 .
  Fact 0.0: 1909 World Series: Pittsburgh Pirates beat Detroit Tigers 8-0 at Bennett Park, Detroit.
  Fact 0.1: 1909 World Series: Rookie Babe Adams tossed a six-hit shutout for his third win of the series.
  Fact 0.2: 1912 World Series: Boston Red Sox beat New York Giants 3-2 in 10 innings (F/10) at Fenway Park, Boston.
  Fact 0.3: 1912 World Series: The decisive contest was actually game eight, as game two had ended in a tie.
  Fact 0.4: 1924 World Series: Washington Senators beat New York Giants 4-3 in 12 innings (F/12) at Griffith Stadium, Washington, D.C.
  Fact 0.5: 1924 World Series: It was the longest game seven in innings.
  Fact 0.6: 1924 World Series: In the 12th inning, Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy.
  Fact 0.7: 1924 World Series: Muddy Ruel scored on a game-winning RBI double by Earl McNeely.
  Fact 0.8: 1925 World Series: Pittsburgh Pirates beat Washington Senators 9-7 at Forbes Field, Pittsburgh.
  Fact 0.9: 1925 World Series: Pittsburgh became the first team in a best-of-seven series to win when trailing three games to one.
  Fact 0.10: 1926 World Series: St. Louis Cardinals beat New York Yankees 3-2 at Yankee Stadium, the Bronx, New York City.
  Fact 0.11: 1926 World Series: Grover Cleveland Alexander struck out Tony Lazzeri with the bases loaded to end the seventh inning and preserve the Cardinals lead.
  Fact 0.12: 1926 World Series: Two innings later, Babe Ruth was caught stealing to end the World Series.
  Fact 0.13: 1931 World Series: St. Louis Cardinals beat Philadelphia Athletics 4-2 at St. Louis, Missouri.
  Fact 0.14: 1931 World Series: It was the last postseason game for the Athletics while based in Philadelphia.
  Fact 0.15: 1934 World Series: St. Louis Cardinals beat Detroit Tigers 11-0 at Navin Field, Detroit.
  Fact 0.16: 1934 World Series: Dizzy Dean's second win, a six-hit shutout on one day of rest, gave the Cardinals the title.
  Fact 0.17: 1940 World Series: Cincinnati Reds beat Detroit Tigers 2-1 at Crosley Field, Cincinnati.
  Fact 0.18: 1940 World Series: The game was over in a snappy 1:47.
  Fact 0.19: 1940 World Series: A sacrifice bunt and a sacrifice fly scored the winning runs.
  Fact 0.20: 1945 World Series: Detroit Tigers beat Chicago Cubs 9-3 at Wrigley Field, Chicago.
  Fact 0.21: 1945 World Series: It was the Cubs' last World Series appearance until 2016.
  Fact 0.22: 1945 World Series: The row has a see-also note for the Curse of the Billy Goat.
  Fact 0.23: 1946 World Series: St. Louis Cardinals beat Boston Red Sox 4-3 at Sportsman's Park III, St. Louis, Missouri.
  Fact 0.24: 1946 World Series: Enos Slaughter's famous mad dash scored the deciding run to win the game and the series.
  Fact 0.25: 1947 World Series: New York Yankees beat Brooklyn Dodgers 5-2 at Yankee Stadium, the Bronx, New York City.
  Fact 0.26: 1947 World Series: This was the Series that featured Cookie Lavagetto's pinch-hit that spoiled Bill Bevens' no-hit bid.
  Fact 0.27: 1947 World Series: Al Gionfriddo's catch robbed Joe DiMaggio of a home run.
  Fact 0.28: 1952 World Series: New York Yankees beat Brooklyn Dodgers 4-2 at Ebbets Field, Brooklyn, New York City.
  Fact 0.29: 1952 World Series: Billy Martin's game-saving infield catch preserved the Yankees lead and helped win the game and the series.
  Fact 0.30: 1955 World Series: Brooklyn Dodgers beat New York Yankees 2-0 at Yankee Stadium, the Bronx, New York City.
  Fact 0.31: 1955 World Series: It was the Dodgers' only championship in Brooklyn.
  Fact 0.32: 1955 World Series: The home team won all games of the series except game seven.
  Fact 0.33: 1956 World Series: New York Yankees beat Brooklyn Dodgers 9-0 at Ebbets Field, Brooklyn, New York City.
  Fact 0.34: 1956 World Series: It was Jackie Robinson's final Major League game.
  Fact 0.35: 1956 World Series: It was the last postseason game at Ebbets Field.
  Fact 0.36: 1956 World Series: It was the Dodgers' last postseason game before moving to Los Angeles.
  Fact 0.37: 1956 World Series: The home team won all games of the series except game seven.
  Fact 0.38: 1957 World Series: Milwaukee Braves beat New York Yankees 5-0 at Yankee Stadium, the Bronx, New York City.
  Fact 0.39: 1957 World Series: It was the Braves' only championship in Milwaukee.
  Fact 0.40: 1958 World Series: New York Yankees beat Milwaukee Braves 6-2 at County Stadium, Milwaukee.
  Fact 0.41: 1958 World Series: The Yankees won their 18th World Series title after being down three games to one.
  Fact 0.42: 1958 World Series: It was the Braves' last postseason game in Milwaukee.
  Fact 0.43: 1960 World Series: Pittsburgh Pirates beat New York Yankees 10-9 at Forbes Field, Pittsburgh.
  Fact 0.44: 1960 World Series: Bill Mazeroski hit the walk-off home run to win the Pirates the championship.
  Fact 0.45: 1960 World Series: It was the last postseason game in Forbes Field.
  Fact 0.46: 1960 World Series: It was the last postseason game for Yankees' manager Casey Stengel.
  Fact 0.47: 1960 World Series: It was the highest scoring game seven.
  Fact 0.48: 1962 World Series: New York Yankees beat San Francisco Giants 1-0 at Candlestick Park, San Francisco.
  Fact 0.49: 1962 World Series: With runners on second and third and two outs, Willie McCovey's line drive to Bobby Richardson ended the game.
  Fact 0.50: 1962 World Series: It was the first game seven that ended with a 1-0 score.
  Fact 0.51: 1964 World Series: St. Louis Cardinals beat New York Yankees 7-5 at Busch Stadium I, St. Louis, Missouri.
  Fact 0.52: 1964 World Series: It was the Yankees' last World Series appearance until 1976.
  Fact 0.53: 1964 World Series: It was the last postseason game at Busch Stadium I.
  Fact 0.54: 1965 World Series: Los Angeles Dodgers beat Minnesota Twins 2-0 at Metropolitan Stadium, Bloomington, Minnesota.
  Fact 0.55: 1965 World Series: Sandy Koufax pitched through immense pain and on two days rest.
  Fact 0.56: 1965 World Series: Sandy Koufax threw a 3-hit shutout to win the Series.
  Fact 0.57: 1965 World Series: Sandy Koufax was named World Series MVP.
  Fact 0.58: 1965 World Series: The home team won all games of the series except game seven.
  Fact 0.59: 1967 World Series: St. Louis Cardinals beat Boston Red Sox 7-2 at Fenway Park, Boston.
  Fact 0.60: 1967 World Series: Bob Gibson wins his third game of the Series.
  Fact 0.61: 1967 World Series: Bob Gibson also homers in this game.
  Fact 0.62: 1968 World Series: Detroit Tigers beat St. Louis Cardinals 4-1 at Busch Stadium II, St. Louis, Missouri.
  Fact 0.63: 1968 World Series: Mickey Lolich out-dueled Bob Gibson.
  Fact 0.64: 1968 World Series: The Tigers rallied from a 3-1 deficit to win the series.
  Fact 0.65: 1968 World Series: It was the first game seven loss by the Cardinals in the World Series.
  Fact 0.66: 1971 World Series: Pittsburgh Pirates beat Baltimore Orioles 2-1 at Memorial Stadium, Baltimore.
  Fact 0.67: 1971 World Series: The home team won all games of the series except game seven.
  Fact 0.68: 1972 World Series: Oakland Athletics beat Cincinnati Reds 3-2 at Riverfront Stadium, Cincinnati.
  Fact 0.69: 1972 World Series: It was the A's first championship in Oakland.
  Fact 0.70: 1972 World Series: The A's were the last American League team to win a World Series game seven on the road until 2017.
  Fact 0.71: 1973 World Series: Oakland Athletics beat New York Mets 5-2 at Oakland-Alameda County Coliseum, Oakland, California.
  Fact 0.72: 1973 World Series: Bert Campaneris and Reggie Jackson hit the only Oakland home runs in this game.
  Fact 0.73: 1973 World Series: Those home runs set the tone for victory and the series.
  Fact 0.74: A row with no year listed: Cincinnati Reds beat Boston Red Sox 4-3 at Fenway Park, Boston.
  Fact 0.75: A row with no year listed: It was one night after Carlton Fisk's famous home run.
  Fact 0.76: A row with no year listed: The Reds rebounded to take the Series.
  Fact 0.77: 1979 World Series: Pittsburgh Pirates beat Baltimore Orioles 4-1 at Memorial Stadium, Baltimore.
  Fact 0.78: 1979 World Series: The Pirates were powered by NLCS and World Series MVP Willie Stargell.
  Fact 0.79: 1979 World Series: The Pirates rallied from a 3-1 deficit to win the series.
  Fact 0.80: 1979 World Series: The Pirates were the last team to win a road game seven until 2014.
  Fact 0.81: 1982 World Series: St. Louis Cardinals beat Milwaukee Brewers 6-3 at Busch Stadium II, St. Louis, Missouri.
  Fact 0.82: 1982 World Series: Joaquín Andújar earned his second win of the Series.
  Fact 0.83: 1982 World Series: Bruce Sutter earned his second save.
  Fact 0.84: 1985 World Series: Kansas City Royals beat St. Louis Cardinals 11-0 at Royals Stadium, Kansas City, Missouri.
  Fact 0.85: 1985 World Series: It was the second World Series held entirely in the state of Missouri.
  Fact 0.86: 1985 Royals also won the 1985 American League Championship Series in seven games.
  Fact 0.87: 1985 Royals became the first team to win the World Series after losing the first two games at home.
  Fact 0.88: 1985 Royals were the last team to come back from 3-1 down to win a title in any of the five major men's professional sports leagues until the Cleveland Cavaliers won the 2016 NBA Finals after trailing 3-1.
  Fact 0.89: 1985 Royals were the last team to win a World Series after trailing 3-1 until the Cubs did so in 2016.
  Fact 0.90: 1986 World Series: New York Mets beat Boston Red Sox 8-5 at Shea Stadium, Flushing, Queens, New York.
  Fact 0.91: 1986 World Series: The Mets became the first team to be one strike away from elimination and come back to win the series.
  Fact 0.92: 1986 World Series: The Mets became the second team to win the series after losing the first two games at home.
  Fact 0.93: A row with no year listed: Minnesota Twins beat St. Louis Cardinals 4-2 at Hubert H. Humphrey Metrodome, Minneapolis.
  Fact 0.94: A row with no year listed: The home team won all seven games of the series for the first time in MLB history.
  Fact 0.95: 1991 World Series: Minnesota Twins beat Atlanta Braves 1-0 in 10 innings (F/10) at Hubert H. Humphrey Metrodome, Minneapolis.
  Fact 0.96: 1991 World Series: Like in 1987, the home team won all games of the series.
  Fact 0.97: 1991 World Series: The Twins won in the bottom of the 10th on a walk-off single by Gene Larkin scoring Dan Gladden after 9 and a fraction scoreless innings of play.
  Fact 0.98: 1991 World Series: Jack Morris went all ten innings for the Twins and was named Series MVP.
  Fact 0.99: 1991 World Series: Twins centerfielder Kirby Puckett hit a famous walk-off home run in the bottom of the eleventh inning of the previous night's game to force this decisive game.
  Fact 0.100: 1997 World Series: Florida Marlins beat Cleveland Indians 3-2 in 11 innings (F/11) at Pro Player Stadium, Miami Gardens, Florida.
  Fact 0.101: 1997 World Series: Florida became the first wild card team to win the World Series.
  Fact 0.102: 1997 World Series: The Marlins rallied for one run in the bottom of the ninth to tie.
  Fact 0.103: 1997 World Series: The Marlins won in the eleventh on an Édgar Rentería RBI single off Charles Nagy.
  Fact 0.104: 2001 World Series: Arizona Diamondbacks beat New York Yankees 3-2 at Bank One Ballpark, Phoenix, Arizona.
  Fact 0.105: 2001 World Series: Arizona became the fastest expansion team to win the World Series.
  Fact 0.106: 2001 World Series: Arizona rallied for two runs in the bottom of the ninth for a come-from-behind win.
  Fact 0.107: 2001 World Series: Luis Gonzalez hit the winning RBI single off closer Mariano Rivera.
  Fact 0.108: 2001 World Series: The home team won every game of the series.
  Fact 0.109: 2002 World Series: Anaheim Angels beat San Francisco Giants 4-1 at Edison International Field, Anaheim, California.
  Fact 0.110: 2002 World Series: Both World Series teams were wild card entrants into the playoffs.
  Fact 0.111: 2002 World Series: John Lackey became the second rookie pitcher to win a World Series game seven.
  Fact 0.112: 2011 World Series: St. Louis Cardinals beat Texas Rangers 6-2 at Busch Stadium, St. Louis, Missouri.
  Fact 0.113: 2011 World Series: The Cardinals became the second team to be one strike away from elimination and come back to win the series.
  Fact 0.114: 2011 World Series: The Cardinals did come back to win the series twice.
  Fact 0.115: 2011 World Series: Just like the 1991 series, a walk-off home run in the bottom of the eleventh inning of game six forced this decisive game.
  Fact 0.116: 2011 World Series: David Freese provided the firepower in that game six walk-off home run.
  Fact 0.117: 2011 World Series: The most recent game seven was won by the home team.
  Fact 0.118: 2014 World Series: San Francisco Giants beat Kansas City Royals 3-2 at Kauffman Stadium, Kansas City, Missouri.
  Fact 0.119: 2014 World Series: Like in 2002, both World Series teams were wild card entrants into the playoffs.
  Fact 0.120: 2014 World Series: The Giants became the first NL team to win a game seven on the road since 1979.
  Fact 0.121: 2014 World Series: Madison Bumgarner was the NLCS MVP and eventual World Series MVP.
  Fact 0.122: 2014 World Series: Madison Bumgarner is normally a starter.
  Fact 0.123: 2014 World Series: Madison Bumgarner pitched five scoreless innings of relief on two days rest.
  Fact 0.124: 2014 World Series: Madison Bumgarner's relief pitching saved the Giants' third title in five years.
  Fact 0.125: 2016 World Series: Chicago Cubs beat Cleveland Indians 8-7 in 10 innings (F/10) at Progressive Field, Cleveland, Ohio.
  Fact 0.126: 2016 World Series: It was the Cubs' first World Series win since 1908.
  Fact 0.127: 2016 World Series: The game was played between the two teams with the longest active championship droughts.
  Fact 0.128: 2016 World Series: The Cubs are the first Major League Baseball team to come back to win the World Series from a 3-1 series deficit since the 1985 Royals.
  Fact 0.129: 2016 World Series: It was the most recent World Series game seven to be decided in extra innings.
  Fact 0.130: 2016 World Series: It was the first extra innings game seven to be won by the road team.
  Fact 0.131: 2017 World Series: Houston Astros beat Los Angeles Dodgers 5-1 at Dodger Stadium, Los Angeles.
  Fact 0.132: 2017 World Series: It was the first instance of back-to-back game sevens since 2001-2002.
  Fact 0.133: 2017 World Series: It was the first World Series championship for the Astros.
  Fact 0.134: 2017 World Series: The Astros became the first AL team to win a World Series game 7 on the road since 1972.
  Fact 0.135: 2017 World Series: The Astros became the second team to win two winner-take-all game sevens in the same postseason since the 1985 Kansas City Royals.

Retrieved items:
<item 1>
Game seven
2012 NLCS San Francisco Giants St. Louis Cardinals 9 -- 0 AT&T Park , San Francisco , California First Giants victory in a winner - take - all game seven . The
Giants would go on to sweep the Detroit Tigers in the World Series . 2017 ALCS Houston Astros New York Yankees 4 -- 0 Minute Maid Park , Houston , Texas The
home team won all games of the series . Houston became the first franchise to win the pennant in both the National League and American League . The Astros would
go on to win the World Series over the Los Angeles Dodgers in seven games , becoming the second MLB team to win two game sevens in a single postseason .
National Hockey League ( edit ) Main article : List of NHL game sevens In the NHL playoffs , game seven can occur in all playoff series . In the Stanley Cup
Final , the NHL gained complete control of the Cup in 1926 , and the league instituted the best - of - seven series starting in the 1938 -- 39 season . No
Stanley Cup Final game seven has ever ended with a 1 -- 0 score .
</item 1>

<item 2>
Game seven
Louis the NL Pennant . The Cardinals would lose the World Series as the Minnesota Twins give new meaning to the term home - field advantage in the World Series
as only the decisive game seven is close . 1988 NLCS Los Angeles Dodgers New York Mets 6 -- 0 Dodger Stadium , Los Angeles Orel Hershiser pitches a five - hit
shutout to send the Dodgers to the World Series . Until 2017 , this was the only game seven of any postseason series to be played in Dodger Stadium . 1991 NLCS
Atlanta Braves Pittsburgh Pirates 4 -- 0 Three Rivers Stadium , Pittsburgh , Pennsylvania Back - to - back shutouts in games six and seven helps the Braves win
the pennant . The Braves would lose the series to Minnesota in seven games in what is regarded as the best World Series of all time . 1992 NLCS Atlanta Braves
Pittsburgh Pirates 3 -- 2 Atlanta -- Fulton County Stadium , Atlanta Atlanta scored all three runs in the bottom of the ninth . Pinch hitter Francisco Cabrera
drove in the two winning runs with a two - out single with the bases loaded . The winning run was scored by Sid Bream .
</item 2>

<item 3>
Game seven
League Championship Series that were decided on a game seven : Year Series Winning team Losing team Score Site Notes 1985 ALCS Kansas City Royals Toronto Blue
Jays 6 -- 2 Exhibition Stadium , Toronto The Royals rallied from a 3 - 1 deficit , including taking the last two on the road to win the pennant . They would go
on to win the 1985 World Series in seven games , becoming the first team in MLB history to win two game sevens in a single postseason . 1986 ALCS Boston Red Sox
California Angels 8 -- 1 Fenway Park , Boston , Massachusetts Dave Henderson 's famous home run off of Donnie Moore turned the tide as Boston rallied from a 3 -
1 deficit to win the series . Boston would lose the 1986 World Series in seven games . NLCS St. Louis Cardinals San Francisco Giants 6 -- 0 Busch Stadium ( II )
, St. Louis , Missouri Although Jeffrey Leonard would homer in the first four games of the series it was not enough as back - to - back shutouts by the
Cardinals in games six and seven gave St. Louis the NL Pennant .
</item 3>

<item 4>
Game seven
Home teams are 19 -- 20 in game sevens . Year Winning team Losing team Score Site Notes 1909 Pittsburgh Pirates Detroit Tigers 8 -- 0 Bennett Park , Detroit
Rookie Babe Adams tosses a six hit shutout for his third win of the series . 1912 Boston Red Sox New York Giants 3 -- 2 F / 10 Fenway Park , Boston This
decisive contest was actually game eight , as game two had ended in a tie . 1924 Washington Senators New York Giants 4 -- 3 F / 12 Griffith Stadium , Washington
, D.C. Longest game seven in innings . In the 12th , Muddy Ruel doubled after a dropped foul popfly by Giants catcher Hank Gowdy . Ruel scored on a game -
winning RBI double by Earl McNeely . 1925 Pittsburgh Pirates Washington Senators 9 -- 7 Forbes Field , Pittsburgh Pittsburgh became the first team in a best of
seven to win the series when trailing three games to one . 1926 St.
</item 4>

<item 5>
2017 World Series
1997 1998 1999 2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series
champions Most Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise )
Broadcasters TV ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series
Category : World Series Fox Major League Baseball Related programs The Cheap Seats ( 2010 -- 2011 ) MLB Whiparound Major League Baseball Game of the Week
Thursday Night Baseball ( 1997 -- 2001 ) This Week in Baseball ( 2000 -- 2011 ) Related articles DirecTV N3D FoxBox FoxTrax Scooter Television contracts ( cable
) MLB Network World Series television ratings National coverage Fox ( 1996 -- present ) FS1 ( 2014 -- present
</item 5>

<item 6>
2017 World Series
Game 5 lasted five hours and seventeen minutes , making it the second longest World Series game in history by time , trailing only the 14 inning contest between
the Astros and the Chicago White Sox in Game 3 of the 2005 World Series in the same ballpark , as well as the third World Series game ever to go beyond five
hours of game time . This game is considered one of the greatest World Series games of all time . Game 6 ( edit ) Kenley Jansen recorded a two - inning save in
Game 6 .
</item 6>

<item 7>
Game seven
Game played between the two teams ( Cleveland Indians and Chicago Cubs ) with the longest active championship droughts . The Cubs are the first Major League
Baseball team to come back to win the World Series from a 3 -- 1 series deficit since the 1985 Royals . Most recent World Series game seven to be decided in
extra innings . First extra innings game seven to be won by the road team . 2017 Houston Astros Los Angeles Dodgers 5 -- 1 Dodger Stadium , Los Angeles First
instance of back - to - back game sevens since 2001 -- 2002 . First World Series championship for the Astros , who became the first AL team to win a World
Series game 7 on the road since 1972 . The Astros became the second team to win two winner - take - all game sevens in the same postseason since the Kansas City
Royals in 1985 . League Championship Series ( edit ) The League Championship Series were originally decided best of five from 1969 until 1984 . Since 1985 when
both were expanded to best of seven contests , 16 series ( 7 American , 9 National ) were decided in game seven .
</item 7>

<item 8>
Game seven
Most codes of football do not employ a best - of - seven series ( or any best - of - x series in general ) , hence game sevens are not played in those leagues .
Some playoff rounds ( such as MLB 's current Division Series ) are played in a best of five format , such that game five has similar qualities to those
described above , though the suspense and drama have less time to build in a shorter series . Furthermore , the World Series of 1903 , 1919 , 1920 , and 1921
were played in a best of nine format , though none of the four went to a decisive game nine . The game seven is comparable to a final or to a single game in a
single - elimination tournament or to a one - game playoff . The game is equivalent to the Super Bowl game in the National Football League in that the game 's
winner is the league 's champion for the season . Contents ( hide ) 1 Examples 1.1 Major League Baseball 1.1. 1 World Series 1.1.
</item 8>

<item 9>
Game seven
1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999
2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most
Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV
ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World
Series Stanley Cup Finals Stanley Cup awarded on challenge basis 1893 --
</item 9>

<item 10>
Game seven
1 World Series 1.1. 2 League Championship Series 1.2 National Hockey League 1.3 National Basketball Association 1.4 Chinese Professional Baseball League 2
Comebacks 3 References Examples ( edit ) Major League Baseball ( edit ) In the MLB Postseason , game seven can occur in the League Championship Series and the
World Series ( the divisional series are best of five ) . In the World Series , there have been 39 decisive game sevens ( of the four World Series that were
contested on a best - of - nine basis , none went to a game nine , though all four included a non-decisive game seven ) . World Series ( edit ) World Series
that were decided on a game seven : All extra inning game sevens involve three franchises each with two appearances , the New York / San Francisco Giants (
losing both ) , the Washington Senators / Minnesota Twins ( winning both ) and the Cleveland Indians ( losing both ) . Home teams are 19 -- 20 in game sevens .
</item 10>
=== END ITEM id=44a589e7b0d3612c ===

=== ITEM id=40257ca18fa4f50a ===
Gold sentences and facts:
Sentence 0: India and Pakistan acquired independence in 1947 , but Governors - General continued to be appointed over each nation until republican constitutions were written . Louis Mountbatten , 1st Earl Mountbatten of Burma remained Governor - General of India for some time after independence , but the two nations were otherwise headed by native Governors - General . India became a secular republic in 1950 ; Pakistan became an Islamic one in 1956 .
  Fact 0.0: India and Pakistan gained independence in 1947.
  Fact 0.1: Governors-General continued to be appointed over each nation after 1947 until republican constitutions were written.
  Fact 0.2: Louis Mountbatten, 1st Earl Mountbatten of Burma, remained Governor-General of India for some time after independence.
  Fact 0.3: The two nations were otherwise headed by native Governors-General.
  Fact 0.4: India became a secular republic in 1950.
  Fact 0.5: Pakistan became an Islamic republic in 1956.

Retrieved items:
<item 1>
Governor-General of India
Lord Mountbatten addressing the Chamber of Princes as Crown Representative in the 1940s The Governor - General originally had power only over the Presidency of
Fort William in Bengal . The Regulating Act , however , granted them additional powers relating to foreign affairs and defence . The other Presidencies of the
East India Company ( Madras , Bombay and Bencoolen ) were not allowed to declare war on or make peace with an Indian prince without receiving the prior approval
of the Governor - General and Council of Fort William . The powers of the Governor - General , in respect of foreign affairs , were increased by the India Act
1784 . The Act provided that the other Governors under the East India Company could not declare war , make peace or conclude a treaty with an Indian prince
unless expressly directed to do so by the Governor - General or by the Company 's Court of Directors . While the Governor - General thus became the controller
of foreign policy in India , he was not the explicit head of British India . That status came only with the Charter Act 1833 , which granted him ``
superintendence , direction and control of the whole civil and military Government '' of all of British India .
</item 1>

<item 2>
Governor-General of India
The title of viceroy was abandoned when British India split into the two independent dominions of India and Pakistan , but the office of governor - general
continued to exist in each country separately -- until they adopted republican constitutions in 1950 and 1956 , respectively . Until 1858 , the governor -
general was selected by the Court of Directors of the East India Company , to whom he was responsible . Thereafter , he was appointed by the sovereign on the
advice of the British government ; the Secretary of State for India , a member of the UK Cabinet , was responsible for instructing him on the exercise of his
powers . After 1947 , the sovereign continued to appoint the governor - general , but did so on the advice of the Indian government . Governors - General served
at the pleasure of the sovereign , though the practice was to have them serve five - year terms . Governors - General could have their commission rescinded ;
and if one was removed , or left , a provisional governor - general was sometimes appointed until a new holder of the office could be chosen .
</item 2>

<item 3>
Governor-General of India
The Chamber of Princes was an institution established in 1920 by a Royal Proclamation of King - Emperor George V to provide a forum in which the princely rulers
could voice their needs and aspirations to the government . The chamber usually met only once a year , with the Viceroy presiding , but it appointed a Standing
Committee , which met more often . Upon independence in August 1947 , the title of Viceroy was abolished . The representative of the British Sovereign became
known once again as the Governor - General . C. Rajagopalachari became the only Indian Governor - General . However , once India acquired independence , the
Governor - General 's role became almost entirely ceremonial , with power being exercised on a day - to - day basis by the Indian cabinet . After the nation
became a republic in 1950 , the President of India continued to perform the same functions .
</item 3>

<item 4>
Governor-General of India
Most Governors - General and Viceroys were peers . Frequently , a Viceroy who was already a peer would be granted a peerage of higher rank , as with the
granting of a marquessate to Lord Reading and an earldom and later a marquessate to Freeman Freeman - Thomas . Of those Viceroys who were not peers , Sir John
Shore was a baronet , and Lord William Bentinck was entitled to the courtesy title ' Lord ' because he was the son of a Duke . Only the first and last Governors
- General -- Warren Hastings and Chakravarti Rajagopalachari -- as well as some provisional Governors - General , had no honorific titles at all . Flag ( edit )
Flag of the Governor General of India ( 1947 -- 50 ) From around 1885 , the Viceroy of India was allowed to fly a Union Flag augmented in the centre with the '
Star of India ' surmounted by a Crown . This flag was not the Viceroy 's personal flag ; it was also used by Governors , Lieutenant Governors , Chief
Commissioners and other British officers in India .
</item 4>

<item 5>
Governor-General of India
The first Governor - General of British India was Warren Hastings , and the first Governor - General of independent India was Louis Mountbatten . Contents (
hide ) 1 History 2 Functions 3 Council 4 Style and title 5 Flag 6 Residence 7 Insignia 8 See also 9 References 10 Further reading History ( edit ) Warren
Hastings , the first Governor - General of Fort William from 1773 to 1785 . Many parts of the Indian subcontinent were governed by the East India Company ,
which nominally acted as the agent of the Mughal Emperor . In 1773 , motivated by corruption in the Company , the British government assumed partial control
over the governance of India with the passage of the Regulating Act of 1773 . A Governor - General and Supreme Council of Bengal were appointed to rule over the
Presidency of Fort William in Bengal . The first Governor - General and Council were named in the Act . The Charter Act 1833 replaced the Governor - General and
Council of Fort William with the Governor - General and Council of India .
</item 5>

<item 6>
Governor-General of India
The officer had direct control only over Fort William , but supervised other British East India Company officials in India . Complete authority over all of
British India was granted in 1833 , and the official came to be known as the `` Governor - General of India '' . In 1858 , the territories of the East India
Company came under the direct control of the British government ; see British Raj . The governor - general ( now also the viceroy ) headed the central
government of India , which administered the provinces of British India , including the Punjab , Bengal , Bombay , Madras , the United Provinces , and others .
However , much of India was not ruled directly by the British government ; outside the provinces of British India , there were hundreds of nominally sovereign
princely states or `` native states '' , whose relationship was not with the British government , but directly with the monarch . To reflect the governor -
general 's role as the representative of the monarch to the feudal rulers of the princely states , from 1858 the term Viceroy and Governor - General of India (
known in short as the Viceroy of India ) was applied to him .
</item 6>

<item 7>
Governor-General of India
Governor - General of India - Wikipedia Governor - General of India Jump to : navigation , search This article is about the position of Governor - General of
India . For individual Governor - Generals , see List of governors - general of India . Viceroy and Governor - General of India Standard of the Governor -
General Louis Mountbatten , the last Viceroy of India & the first Governor - General during the dominion period Style His Excellency Residence Viceroy 's House
Appointer East India Company ( to 1858 ) Monarch of India ( from 1858 ) Formation 20 October 1774 First holder Warren Hastings Final holder Chakravarthi
Rajagopalachari Abolished 26 January 1950 The Governor - General of India ( or , from 1858 to 1947 , officially the Viceroy and Governor - General of India ,
commonly shortened to Viceroy of India ) was originally the head of the British administration in India and , later , after Indian independence in 1947 , the
representative of the Indian head of state . The office was created in 1773 , with the title of Governor - General of the Presidency of Fort William .
</item 7>

<item 8>
Governor-General of India
The power to elect the Governor - General was retained by the Court of Directors , but the choice became subject to the Sovereign 's approval . After the Indian
Rebellion of 1857 , the East India Company 's territories in India were put under the direct control of the Sovereign . The Government of India Act 1858 vested
the power to appoint the Governor - General in the Sovereign . The Governor - General , in turn , had the power to appoint all lieutenant governors in India ,
subject to the Sovereign 's approval . India and Pakistan acquired independence in 1947 , but Governors - General continued to be appointed over each nation
until republican constitutions were written . Louis Mountbatten , 1st Earl Mountbatten of Burma remained Governor - General of India for some time after
independence , but the two nations were otherwise headed by native Governors - General . India became a secular republic in 1950 ; Pakistan became an Islamic
one in 1956 . Functions ( edit ) Lord Curzon in his robes as Viceroy of India , a post he held from 1899 to 1905 .
</item 8>

<item 9>
Governor-General of India
Throughout the British administration , Governors - General retreated to the Viceregal Lodge ( Rashtrapati Niwas ) at Shimla each summer to escape the heat ,
and the government of India moved with them . The Viceregal Lodge now houses the Indian Institute of Advanced Study . Insignia ( edit ) See also : Star of India
( flag ) Badge of the Governor - General ( 1885 -- 1947 ) Standard of the Governor - General ( 1885 -- 1947 ) Standard of the Governor - General ( 1947 -- 50 )
See also ( edit ) India portal Pakistan portal United Kingdom portal Politics portal List of governors - general of India Commander - in - Chief , India British
Empire Emperor of India Indian independence movement Council of India British Raj Secretary of State for India India Office Indian Civil Service Partition of
India History of Bangladesh History of India History of Pakistan References ( edit ) Jump up ^ The term British India is mistakenly used to mean the same as the
British Indian Empire , which included both the provinces and the Native States . Jump up ^ `` Imperial Impressions '' . Hindustan Times . 20 July 2011 .
</item 9>

<item 10>
Governor-General of India
The Act also granted legislative powers to the Governor - General and Council . After 1858 , the Governor - General ( now usually known as the Viceroy )
functioned as the chief administrator of India and as the Sovereign 's representative . India was divided into numerous provinces , each under the head of a
Governor , Lieutenant Governor or Chief Commissioner or Administrator . Governors were appointed by the British Government , to whom they were directly
responsible ; Lieutenant Governors , Chief Commissioners , and Administrators , however , were appointed by and were subordinate to the Viceroy . The Viceroy
also oversaw the most powerful princely rulers : the Nizam of Hyderabad , the Maharaja of Mysore , the Maharaja ( Scindia ) of Gwalior , the Maharaja of Jammu
and Kashmir and the Gaekwad ( Gaekwar ) Maharaja of Baroda . The remaining princely rulers were overseen either by the Rajputana Agency and Central India Agency
, which were headed by representatives of the Viceroy , or by provincial authorities .
</item 10>
=== END ITEM id=40257ca18fa4f50a ===

=== ITEM id=ac2bc02dbe573f29 ===
Gold sentences and facts:
Sentence 0: `` Rip It Up '' was a 1983 single by Scottish post-punk band Orange Juice . It was the second single to be released from their 1982 album of the same name . The song became the band 's only UK top 40 success , reaching no . 8 in the chart . `` Rip It Up '' signalled a departure from the sound of the band 's earlier singles , with Chic - influenced guitars and using a synthesiser to create a more disco - oriented sound .
  Fact 0.0: 'Rip It Up' was a 1983 single by Scottish post-punk band Orange Juice.
  Fact 0.1: 'Rip It Up' was the second single released from Orange Juice's 1982 album of the same name.
  Fact 0.2: 'Rip It Up' became Orange Juice's only UK top 40 success.
  Fact 0.3: 'Rip It Up' reached no. 8 in the chart.
  Fact 0.4: 'Rip It Up' signalled a departure from the sound of the band's earlier singles.
  Fact 0.5: 'Rip It Up' had Chic-influenced guitars.
  Fact 0.6: 'Rip It Up' used a synthesiser to create a more disco-oriented sound.
Sentence 1: `` Rip It Up '' Single by Orange Juice from the album Rip It Up B - side `` Snake Charmer '' `` A Sad Lament '' Released February 1983 Format 7 '' 12 '' Recorded Berwick Street Studios , London Genre Post-punk Label Polydor Songwriter ( s ) Edwyn Collins Producer ( s ) Martin Hayles Orange Juice singles chronology `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 ) `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 )
  Fact 1.0: 'Rip It Up' is a single by Orange Juice from the album Rip It Up.
  Fact 1.1: The B-side of 'Rip It Up' is 'Snake Charmer'.
  Fact 1.2: The B-side of 'Rip It Up' is 'A Sad Lament'.
  Fact 1.3: 'Rip It Up' was released in February 1983.
  Fact 1.4: 'Rip It Up' was released in 7-inch and 12-inch formats.
  Fact 1.5: 'Rip It Up' was recorded at Berwick Street Studios, London.
  Fact 1.6: 'Rip It Up' is in the post-punk genre.
  Fact 1.7: 'Rip It Up' was released on the Polydor label.
  Fact 1.8: 'Rip It Up' was songwritten by Edwyn Collins.
  Fact 1.9: 'Rip It Up' was produced by Martin Hayles.
  Fact 1.10: Orange Juice's singles chronology lists 'I Ca n't Help Myself' (1982), 'Rip It Up' (1983) and 'Flesh of My Flesh' (1983).

Retrieved items:
<item 1>
Rip It Up (Orange Juice song)
Official Charts Company . Retrieved 26 May 2013 . External links ( edit ) Music video on YouTube ( hide ) Orange Juice Edwyn Collins James Kirk David McClymont
Steven Daly Malcolm Ross Zeke Manyika Studio albums You Ca n't Hide Your Love Forever Rip It Up Texas Fever The Orange Juice Singles `` Falling and Laughing ''
`` Rip It Up '' Compilations Ostrich Churchyard The Heather 's on Fire The Glasgow School Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Rip_It_Up_(Orange_Juice_song)&oldid=751734896 '' Categories : 1983 singles Orange Juice ( band ) songs 1983 songs
Polydor Records singles Songs written by Edwyn Collins Hidden categories : Use dmy dates from May 2015 Use British English from May 2015 All articles with
unsourced statements Articles with unsourced statements from August 2013 Track listings with deprecated parameters Talk Contents About Wikipedia Add links This
page was last edited on 27 November 2016 , at 15 : 08 . About Wikipedia
</item 1>

<item 2>
Rip It Up (Orange Juice song)
8 in the chart . `` Rip It Up '' signalled a departure from the sound of the band 's earlier singles , with Chic - influenced guitars and using a synthesiser to
create a more disco - oriented sound . The song was sampled in 2009 by British soul singer Beverley Knight on her song `` In Your Shoes '' from the album 100 %
. In 2014 , NME ranked it at number 216 in its list of The 500 Greatest Songs of All Time . It was also included by Pitchfork at number 157 in a list of The
Best 200 Songs of the 1980s . Contents ( hide ) 1 Recording and influences 2 Music video 3 UK single release 3.1 Track listing 4 Chart positions 5 References 6
External links Recording and influences ( edit ) Roland TB - 303 The song was recorded as part of the sessions for Orange Juice 's second studio album and would
go on to become the title track of said album .
</item 2>

<item 3>
Rip It Up (Orange Juice song)
`` Rip It Up '' ( Long Version ) Edwyn Collins 2 . `` A Sad Lament '' ( Long Version ) Edwyn Collins Chart positions ( edit ) Chart ( 1983 ) Peak position UK
Singles Chart 8 New Zealand Singles Chart 42 References ( edit ) Jump up ^ http://www.rocklistmusic.co.uk/nme_500_greatest_songs_2014.htm Jump up ^
http://pitchfork.com/features/lists-and-guides/9700-the-200-best-songs-of-the-1980s/?page=3 ^ Jump up to : `` Buzzcocks : Boredom / Orange Juice : Rip It Up ''
. Stylus Magazine . 25 August 2004 . Retrieved 26 May 2013 . ^ Jump up to : `` Orange Juice - Rip It Up 12 '' . Discogs . Retrieved 26 May 2013 . Jump up ^ ``
Orange Juice - Rip It Up '' . Discogs . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice - Rip It Up '' . Discogs . Retrieved 26 May 2013 .
</item 3>

<item 4>
Rip It Up (Orange Juice song)
Discogs . Retrieved 26 May 2013 . ^ Jump up to : `` Orange Juice - Rip It Up ( Song ) '' . charts.org.nz . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice -
Rip It Up '' . Discogs . Retrieved 26 May 2013 . Jump up ^ `` Orange Juice '' . Official Charts Company . Retrieved 26 May 2013 .
</item 4>

<item 5>
Rip It Up (Orange Juice song)
Track listing ( edit ) Single 7 '' No . Title Writer ( s ) Length 1 . `` Rip It Up '' Edwyn Collins 3 : 51 2 . `` Snake Charmer '' Malcolm Ross 4 : 43 Double 7
'' No . Title Writer ( s ) Length 1 . `` Rip It Up '' Edwyn Collins 2 . `` Snake Charmer '' Malcolm Ross 3 . `` Love Sick '' ( Live ) Edwyn Collins 4 . `` A Sad
Lament '' Edwyn Collins 12 '' No . Title Writer ( s ) Length 1 . `` Rip It Up '' ( Long Version ) Edwyn Collins 2 .
</item 5>

<item 6>
Rip It Up (Orange Juice song)
It marked a departure from their previous guitar - pop based material , instead utilising Chic style guitar - funk and a bubbling Roland TB - 303 synthesiser
bassline , becoming the first chart single to feature the instrument . The song also features a snatch of the guitar riff from `` Boredom '' , a song by
Buzzcocks that featured on their debut Spiral Scratch EP . The riff chimes briefly in , just as Collins namechecks the song in the lyrics claiming that `` ...
and my favourite song is entitled ' Boredom ' . '' Backing vocals on the song were provided by Paul Quinn , the lead singer of fellow Scottish band Bourgie
Bourgie , with whom Collins would later record a single in 1984 , a cover of the Velvet Underground song `` Pale Blue Eyes . '' Music video ( edit ) The video
opens with the band in a futuristic , but cheaply constructed , control room as they sing , dance and operate various controls . The band then watch themselves
on a monitor screen as they walk down a rainy British high street dressed in incongruous , brightly coloured summer clothes .
</item 6>

<item 7>
Rip It Up (Orange Juice song)
The video then cuts back to the control room , this time with the band playing their instruments superimposed over it , before returning to more scenes of a
British city in torrential rain as the band walk around in scuba diving gear . The video finally cuts back to the band playing in a silver foil covered room ,
before superimposing them over a pile of random photographs . UK single release ( edit ) `` Rip It Up '' was released as a single in the UK in February 1983 .
The seven inch vinyl version of the single was available in three versions , a double pack including a second seven - inch and a fold out poster , along with
two versions of the standard release , initially with a silver injection moulded labels , and then subsequently with paper printed labels . The song was also
released on twelve inch vinyl , with extended versions of the title track and B - side . All versions were housed in a paper sleeve depicting a US P - 40
Warhawk fighter plane ( decorated with eyes and teeth ) partially submerged , tail first , in the sea , drawn by Edwyn Collins . Track listing ( edit ) Single 7
'' No .
</item 7>

<item 8>
Total Eclipse of the Heart
Video performances have also been released on Tyler 's DVDs , Bonnie on Tour ( 2006 ) and the DVD edition of Live in Germany 1993 . Credits and personnel ( edit
) Credits adapted from AllMusic ( from the album , Faster Than the Speed of Night ) : Bonnie Tyler -- lead vocals Rick Derringer -- guitar Steve Buslowe -- bass
guitar Roy Bittan -- piano Larry Fast -- synthesizers Steve Margoshes -- additional synths Max Weinberg -- drums Jimmy Maelen -- percussion Holly Sherwood --
lead backing vocals Rory Dodd -- `` Turn around '' and backing vocals Eric Troyer -- backing vocals Other versions by Bonnie Tyler ( edit ) Since the original
release in 1983 , Tyler has re-recorded the song several times for albums and subsequent single releases . Her first and most successful re-recording of the
song was released in 2003 . Tyler recorded a French / English duet version called `` Si demain ... ( Turn Around ) '' with Kareen Antonn . It peaked at number
one in France and Belgium . Tyler released another version of the song in 2004 as a duet with Peter Brocklehurst on his album For You .
</item 8>

<item 9>
Rip It Up (Orange Juice song)
Rip It Up ( Orange Juice song ) - wikipedia Rip It Up ( Orange Juice song ) Jump to : navigation , search `` Rip It Up '' Single by Orange Juice from the album
Rip It Up B - side `` Snake Charmer '' `` A Sad Lament '' Released February 1983 Format 7 '' 12 '' Recorded Berwick Street Studios , London Genre Post-punk
Label Polydor Songwriter ( s ) Edwyn Collins Producer ( s ) Martin Hayles Orange Juice singles chronology `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' (
1983 ) `` Flesh of My Flesh '' ( 1983 ) `` I Ca n't Help Myself '' ( 1982 ) `` Rip It Up '' ( 1983 ) `` Flesh of My Flesh '' ( 1983 ) `` Rip It Up '' was a 1983
single by Scottish post-punk band Orange Juice . It was the second single to be released from their 1982 album of the same name . The song became the band 's
only UK top 40 success , reaching no . 8 in the chart .
</item 9>

<item 10>
Freaky Friday (song)
`` Get Back Up '' `` Ai n't Thinkin ' ' Bout You '' `` Champion '' `` My Last '' `` One Night Stand '' `` Best Love Song '' `` Pot of Gold '' `` Body 2 Body ''
`` Better with the Lights Off '' `` International Love '' `` Another Round '' `` Why Stop Now '' `` Birthday Cake ( Remix ) '' `` Right by My Side '' `` Take It
to the Head '' `` I Can Only Imagine '' `` Put It Down '' `` Algo Me Gusta de Ti '' `` Celebration '' `` Ready '' `` As Your Friend '' `` For the Road '' ``
Beat It '' `` Sweet Serenade '' `` Show Me '' `` Main Chick '' `` Hold You Down '' `` Only '' `` Post to Be '' `` Waves ( Robin Schulz Remix ) '' `` Hotel '' ``
Private Show '' `` Five More Hours '' `` You Changed Me '' `` Do It Again '' `` How Many Times '' `` Fun '' `` All Eyes on You '' `` Body on Me '' `` Moses ''
`` Play No Games '' `` Player '' `` Sorry '' `` Gold Slugs '' `` Paradise '' `` Drifting '' `` Wishing '' `` Something New '' `` I 'm the Man ( Remix ) '' `` No
Romeo No
</item 10>
=== END ITEM id=ac2bc02dbe573f29 ===

=== ITEM id=e13fa25f74b056dc ===
Gold sentences and facts:
Sentence 0: Coordinates : 90 ° N 0 ° W ﻿ / ﻿ 90 ° N - 0 ° E ﻿ / 90 ; - 0
  Fact 0.0: The coordinates are listed as 90 degrees N, 0 degrees W.
  Fact 0.1: The coordinates are listed as 90 degrees N, 0 degrees E.
  Fact 0.2: The coordinates are listed as 90, -0.
Sentence 1: The North Pole is the northernmost point on the Earth , lying diametrically opposite the South Pole . It defines geodetic latitude 90 ° North , as well as the direction of true north . At the North Pole all directions point south ; all lines of longitude converge there , so its longitude can be defined as any degree value . Along tight latitude circles , counterclockwise is east and clockwise is west . The North Pole is at the center of the Northern Hemisphere .
  Fact 1.0: The North Pole is the northernmost point on the Earth.
  Fact 1.1: The North Pole lies diametrically opposite the South Pole.
  Fact 1.2: The North Pole defines geodetic latitude 90 degrees North.
  Fact 1.3: The North Pole defines the direction of true north.
  Fact 1.4: At the North Pole all directions point south.
  Fact 1.5: At the North Pole all lines of longitude converge.
  Fact 1.6: The longitude of the North Pole can be defined as any degree value.
  Fact 1.7: Along tight latitude circles, counterclockwise is east.
  Fact 1.8: Along tight latitude circles, clockwise is west.
  Fact 1.9: The North Pole is at the center of the Northern Hemisphere.

Retrieved items:
<item 1>
Graph query row:
entity: North Pole
kind: pole
relationship: LOCATED_IN
related: Arctic Ocean
detail: None
</item 1>

<item 2>
North Pole
Exploration ( edit ) See also : Arctic exploration , Farthest North , List of Arctic expeditions , and List of firsts in the Geographic North Pole Pre-1900 (
edit ) Gerardus Mercator 's map of the North Pole from 1595 C.G. Zorgdragers map of the North Pole from 1720 As early as the 16th century , many prominent
people correctly believed that the North Pole was in a sea , which in the 19th century was called the Polynya or Open Polar Sea . It was therefore hoped that
passage could be found through ice floes at favorable times of the year . Several expeditions set out to find the way , generally with whaling ships , already
commonly used in the cold northern latitudes . One of the earliest expeditions to set out with the explicit intention of reaching the North Pole was that of
British naval officer William Edward Parry , who in 1827 reached latitude 82 ° 45 ′ North . In 1871 the Polaris expedition , a US attempt on the Pole led by
Charles Francis Hall , ended in disaster .
</item 2>

<item 3>
North Pole
Owing to its remoteness , the Pole is sometimes identified with a mysterious mountain of ancient Iranian tradition called Mount Qaf ( Jabal Qaf ) , the ``
farthest point of the earth '' . According to certain authors , the Jabal Qaf of Muslim cosmology is a version of Rupes Nigra , a mountain whose ascent , like
Dante 's climbing of the Mountain of Purgatory , represents the pilgrim 's progress through spiritual states . In Iranian theosophy , the heavenly Pole , the
focal point of the spiritual ascent , acts as a magnet to draw beings to its `` palaces ablaze with immaterial matter . '' See also ( edit ) Arctic portal
Arctic cooperation and politics Arctic Council Biome Celestial pole Ecliptic pole Inuit Circumpolar Council North Pole , Alaska Polaris Poles of astronomical
bodies Willem Barentsz Notes ( edit ) A.
</item 3>

<item 4>
North Pole
Finding the North Pole . The Lyons press . ISBN 1 - 59228 - 072 - 2 . CS1 maint : Multiple names : authors list ( link ) Pal , Christopher ( 2002 ) . The Oddest
Place on Earth : Rediscovering the North Pole . Writer 's Showcase . ISBN 0 - 595 - 21454 - 1 . Fairfield , William ( 1885 ) . Paradise Found a Cradle of the
Human Race at the North Pole a Study of the Prehistoric World . Houghton , Mifflin . ISBN 0 - 665 - 37759 - 2 . External links ( edit ) Wikimedia Commons has
media related to North Pole . Arctic Council The Northern Forum North Pole Web Cam FAQ on the Arctic and the North Pole Daylight , Darkness and Changing of the
Seasons at the North Pole Video of the Nuclear Icebreaker Yamal visiting the North Pole in 2001 Polar Discovery : North Pole Observatory Expedition Polar
exploration Arctic Ocean History Expeditions Research stations Farthest North North Pole Barentsz Hudson Marmaduke Carolus Parry North Magnetic Pole J. Ross
J.C.
</item 4>

<item 5>
South Pole
The geographic coordinates of the South Pole are usually given simply as 90 ° S , since its longitude is geometrically undefined and irrelevant . When a
longitude is desired , it may be given as 0 ° . At the South Pole , all directions face north . For this reason , directions at the Pole are given relative to
`` grid north '' , which points northwards along the prime meridian . Along tight latitude circles , clockwise is east , and counterclockwise is west , opposite
to the North Pole . The Geographic South Pole is located on the continent of Antarctica ( although this has not been the case for all of Earth 's history
because of continental drift ) . It sits atop a featureless , barren , windswept and icy plateau at an altitude of 2,835 metres ( 9,301 ft ) above sea level ,
and is located about 1,300 km ( 800 mi ) from the nearest open sea at Bay of Whales . The ice is estimated to be about 2,700 metres ( 9,000 ft ) thick at the
Pole , so the land surface under the ice sheet is actually near sea level .
</item 5>

<item 6>
North Pole
^ Data is from a Greenlandic Weather Station at 83 ° 38 ′ N 033 ° 22 ′ W ﻿ / ﻿ 83.633 ° N 33.367 ° W ﻿ / 83.633 ; - 33.367 ﻿ ( Greenlandic Weather Station )
located 709 km ( 441 mi ) from the North Pole . References ( edit ) Jump up ^ Black , Richard ( 8 April 2001 ) . New warning on Arctic sea ice melt . BBC Jump
up ^ Ljunggren , David ( 5 March 2009 ) . Arctic summer ice could vanish by 2013 : expert . Reuters Jump up ^ Russian sub plants flag at North Pole , Reuters ,
2 August 2007 Jump up ^ Андерсон , Уильям Роберт ( 1965 ) . `` '' Наутилус '' у Северного полюса `` . Воениздат . Retrieved 12 January 2012 . Jump up ^ Mouton ,
M.W. ( 1968 ) . The International Regime of the Polar Regions .
</item 6>

<item 7>
North Pole
North Pole - wikipedia North Pole Jump to : navigation , search For other uses , see North Pole ( disambiguation ) . Not to be confused with North Magnetic Pole
. Coordinates : 90 ° N 0 ° W ﻿ / ﻿ 90 ° N - 0 ° E ﻿ / 90 ; - 0 An azimuthal projection showing the Arctic Ocean and the North Pole . The map also shows the 75th
parallel north and 60th parallel north . Sea ice at the North Pole in 2006 The North Pole , also known as the Geographic North Pole or Terrestrial North Pole ,
is ( subject to the caveats explained below ) defined as the point in the Northern Hemisphere where the Earth 's axis of rotation meets its surface . The North
Pole is the northernmost point on the Earth , lying diametrically opposite the South Pole . It defines geodetic latitude 90 ° North , as well as the direction
of true north . At the North Pole all directions point south ; all lines of longitude converge there , so its longitude can be defined as any degree value .
Along tight latitude circles , counterclockwise is east and clockwise is west .
</item 7>

<item 8>
North Pole
Along tight latitude circles , counterclockwise is east and clockwise is west . The North Pole is at the center of the Northern Hemisphere . While the South
Pole lies on a continental land mass , the North Pole is located in the middle of the Arctic Ocean amid waters that are almost permanently covered with
constantly shifting sea ice . This makes it impractical to construct a permanent station at the North Pole ( unlike the South Pole ) . However , the Soviet
Union , and later Russia , constructed a number of manned drifting stations on a generally annual basis since 1937 , some of which have passed over or very
close to the Pole . Since 2002 , the Russians have also annually established a base , Barneo , close to the Pole . This operates for a few weeks during early
spring . Studies in the 2000s predicted that the North Pole may become seasonally ice - free because of Arctic ice shrinkage , with timescales varying from 2016
to the late 21st century or later .
</item 8>

<item 9>
North Pole
2 MLAE 2009 Expedition 2.4. 3 MLAE 2013 Expedition 3 Day and night 4 Time 5 Climate 6 Flora and fauna 7 Territorial claims to the North Pole and Arctic regions
8 Cultural associations 9 See also 10 Notes 11 References 12 Further reading 13 External links Precise definition ( edit ) See also : Polar motion The Earth 's
axis of rotation -- and hence the position of the North Pole -- was commonly believed to be fixed ( relative to the surface of the Earth ) until , in the 18th
century , the mathematician Leonhard Euler predicted that the axis might `` wobble '' slightly . Around the beginning of the 20th century astronomers noticed a
small apparent `` variation of latitude , '' as determined for a fixed point on Earth from the observation of stars . Part of this variation could be attributed
to a wandering of the Pole across the Earth 's surface , by a range of a few metres . The wandering has several periodic components and an irregular component .
</item 9>

<item 10>
North Pole
There is no permanent human presence at the North Pole and no particular time zone has been assigned . Polar expeditions may use any time zone that is
convenient , such as Greenwich Mean Time , or the time zone of the country from which they departed . Climate ( edit ) Arctic ice shrinkages of 2007 compared to
2005 and also compared to the 1979 -- 2000 average . Main article : Climate of the Arctic See also : Climate change in the Arctic The North Pole is
substantially warmer than the South Pole because it lies at sea level in the middle of an ocean ( which acts as a reservoir of heat ) , rather than at altitude
on a continental land mass . Despite being an ice cap , it shares some characteristics with a tundra climate ( ETf ) due to the July and August temperatures
peaking just above freezing . Winter temperatures at the North Pole can range from about − 50 to − 13 ° C ( − 58 to 9 ° F ) , averaging around − 31 ° C ( − 24 °
F ) .
</item 10>
=== END ITEM id=e13fa25f74b056dc ===
