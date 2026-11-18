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

This file has 7 items.

=== ITEM id=a7927d1cef428dd7 ===
Gold sentences and facts:
Sentence 0: Polar moment of inertia is a quantity used to predict an object 's ability to resist torsion , in objects ( or segments of objects ) with an invariant circular cross section and no significant warping or out - of - plane deformation . It is used to calculate the angular displacement of an object subjected to a torque . It is analogous to the area moment of inertia , which characterizes an object 's ability to resist bending and is required to calculate displacement .
  Fact 0.0: Polar moment of inertia is a quantity used to predict an object's ability to resist torsion.
  Fact 0.1: Polar moment of inertia predicts torsion resistance for objects, or segments of objects, with an invariant circular cross section.
  Fact 0.2: Polar moment of inertia predicts torsion resistance for objects with no significant warping or out-of-plane deformation.
  Fact 0.3: Polar moment of inertia is used to calculate the angular displacement of an object subjected to a torque.
  Fact 0.4: Polar moment of inertia is analogous to the area moment of inertia.
  Fact 0.5: The area moment of inertia characterizes an object's ability to resist bending.
  Fact 0.6: The area moment of inertia is required to calculate displacement.

Retrieved items:
<item 1>
Polar moment of inertia
In such cases , the torsion constant can be substituted instead . In objects with significant cross-sectional variation ( along the axis of the applied torque )
, which can not be analyzed in segments , a more complex approach may have to be used . See 3 - D elasticity . However the polar moment of area can be used to
calculate the moment of inertia of an object with arbitrary cross-section . Definition ( edit ) A schematic showing how the polar moment of area is calculated
for an arbitrary shape about an axis o. ρ is the radial distance to the element dA .
</item 1>

<item 2>
Polar moment of inertia
= π D 4 32 ( \ displaystyle I_ ( z ) = ( \ frac ( \ pi D ^ ( 4 ) ) ( 32 ) ) ) Mass moment of inertia J z = I z ρ l = π ρ l D 4 32 ( \ displaystyle J_ ( z ) = I_
( z ) \ rho l = ( \ frac ( \ pi \ rho lD ^ ( 4 ) ) ( 32 ) ) ) where : d ( \ displaystyle d ) is the inner diameter in m D ( \ displaystyle D ) is the outside
diameter in m J z ( \ displaystyle J_ ( z ) ) is the mass moment of inertia in kg m l ( \ displaystyle l ) is the length of cylinder in m ρ ( \ displaystyle \
rho ) is the specific mass in kg / m See also ( edit ) Torsion constant Torsion spring Torsion bar Area moment of inertia References ( edit ) Jump up ^ Ugural
AC , Fenster SK . Advanced Strength and Applied Elasticity . 3rd Ed . Prentice - Hall Inc . Englewood Cliffs , NJ . 1995 .
</item 2>

<item 3>
Polar moment of inertia
Moment of inertia is , strictly , the second moment of mass with respect to distance from an axis : I = ∫ m r 2 d m ( \ displaystyle I = \ int _ ( m ) r ^ ( 2 )
\ mathrm ( d ) m ) , where r is the distance to some potential rotation axis , and the integral is over all the infinitesimal elements of mass , dm , in a three
- dimensional space occupied by an object . The MOI , in this sense , is the analog of mass for rotational problems . Contents ( hide ) 1 Limitations 2
Definition 3 Unit 4 Conversion from Area Moment of Inertia 5 Application 6 Sample calculation 7 Comparing various moments of inertias for a cylinder 7.1 Polar
moment of inertia 8 See also 9 References 10 External links Limitations ( edit ) The polar moment of area can not be used to analyze shafts with non-circular
cross-sections . In such cases , the torsion constant can be substituted instead .
</item 3>

<item 4>
Polar moment of inertia
Englewood Cliffs , NJ . 1995 . ISBN 0 - 13 - 137589 - X . Jump up ^ http://www.efunda.com/math/areas/MomentOfInertia.cfm Jump up ^ Blodgeit . Design of Welded
Structures ( 1982 ) . Section 2.10. 2 . Jump up ^ `` What is the difference between the Polar Moment of Inertia , IPIP and the torsional constant , JTJT of a
cross section ? '' .
</item 4>

<item 5>
Polar moment of inertia
If one adds a factor of safety of 5 and re-calculates the radius with the maximal stress equal to the yield stress / 5 the result is a radius of 0.343 m , or a
diameter of 69 cm , the approximate size of a turboset shaft in a nuclear power plant . Comparing various moments of inertias for a cylinder ( edit ) Polar
moment of inertia ( edit ) Hollow cylinder I z = π ( D 4 − d 4 ) 32 ( \ displaystyle I_ ( z ) = ( \ frac ( \ pi \ left ( D ^ ( 4 ) - d ^ ( 4 ) \ right ) ) ( 32
) ) ) Mass moment of inertia J z = I z ρ l = π ρ l ( D 4 − d 4 ) 32 ( \ displaystyle J_ ( z ) = I_ ( z ) \ rho l = ( \ frac ( \ pi \ rho l \ left ( D ^ ( 4 ) -
d ^ ( 4 ) \ right ) ) ( 32 ) ) ) Solid cylinder I z = π D 4 32 ( \ displaystyle I_ ( z ) = ( \ frac
</item 5>

<item 6>
Polar moment of inertia
Conversion from area moment of inertia ( edit ) By the perpendicular axis theorem , the following equation relates I to the area moments of inertia about the
other two mutually perpendicular axes : I z = I x + I y ( \ displaystyle I_ ( z ) = I_ ( x ) + I_ ( y ) ) I x = I y ( \ displaystyle I_ ( x ) = I_ ( y ) ) is
only true for a circular or hollow circular section with a constant thickness . For non circular cross section analytical formulas are needed : torsion constant
. Second moment of inertia , I , is the resistance of a solid to being twisted . Polar moment of inertia , I or J , is the inertia moment of a rotating solid
Application ( edit ) The polar moment of area appears in the formulae that describe torsional stress and angular displacement .
</item 6>

<item 7>
Polar moment of inertia
Polar moment of inertia - wikipedia Polar moment of inertia Jump to : navigation , search Polar moment of inertia is a quantity used to predict an object 's
ability to resist torsion , in objects ( or segments of objects ) with an invariant circular cross section and no significant warping or out - of - plane
deformation . It is used to calculate the angular displacement of an object subjected to a torque . It is analogous to the area moment of inertia , which
characterizes an object 's ability to resist bending and is required to calculate displacement . The larger the polar moment of area , the less the beam will
twist , when subjected to a given torque . Polar moment of area should not be confused with moment of inertia , which characterizes an object 's angular
acceleration due to a torque . See moment ( physics ) .
</item 7>

<item 8>
Polar moment of inertia
. first1 = missing last1 = in Authors list ( help ) External links ( edit ) Torsion of Shafts - engineeringtoolbox.com Elastic Properties and Young Modulus for
some Materials - engineeringtoolbox.com Retrieved from `` https://en.wikipedia.org/w/index.php?title=Polar_moment_of_inertia&oldid=805920942 '' Categories :
Continuum mechanics Mechanics Hidden categories : CS1 errors : missing author or editor Talk Contents About Wikipedia Eesti 한국어 עברית Magyar Русский Simple
English Українська 中文 Edit links This page was last edited on 18 October 2017 , at 13 : 57 . About Wikipedia
</item 8>

<item 9>
Polar moment of inertia
See moment ( physics ) . Note : It has become common to use moment of inertia ( MOI ) to refer to either or both of the planar second moment of area , I = ∫ A x
2 d A ( \ displaystyle I = \ int _ ( A ) x ^ ( 2 ) \ , \ mathrm ( d ) A ) , where x is the distance to some reference plane , or the polar second moment of area
, I = ∫ A r 2 d A ( \ displaystyle I = \ int _ ( A ) r ^ ( 2 ) \ , \ mathrm ( d ) A ) , where r is the distance to some reference axis . In each case the
integral is over all the infinitesimal elements of area , dA , in some two - dimensional cross-section .
</item 9>

<item 10>
Polar moment of inertia
I z = ∫ A ρ 2 d A ( \ displaystyle I_ ( z ) = \ int _ ( A ) \ rho ^ ( 2 ) \ , dA ) I = the polar moment of area about the axis z dA = an elemental area ρ = the
radial distance to the element dA from the axis z For a circular section with radius r : I z = ∫ 0 2 π ∫ 0 r ρ 2 ρ d ρ d φ = π r 4 2 ( \ displaystyle I_ ( z ) =
\ int _ ( 0 ) ^ ( 2 \ pi ) \ int _ ( 0 ) ^ ( r ) \ rho ^ ( 2 ) \ rho \ , d \ rho \ , d \ phi = ( \ frac ( \ pi r ^ ( 4 ) ) ( 2 ) ) ) Unit ( edit ) The SI unit
for polar moment of area , like the area moment of inertia , is meter to the fourth power ( m ) .
</item 10>
=== END ITEM id=a7927d1cef428dd7 ===

=== ITEM id=1c27f151661c0b0d ===
Gold sentences and facts:
Sentence 0: Ballets by Pyotr Ilyich Tchaikovsky Swan Lake ( 1876 ) Sleeping Beauty ( 1889 ) The Nutcracker ( 1892 ) List of all compositions
  Fact 0.0: The list is titled Ballets by Pyotr Ilyich Tchaikovsky.
  Fact 0.1: Swan Lake, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1876.
  Fact 0.2: Sleeping Beauty, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1889.
  Fact 0.3: The Nutcracker, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1892.
  Fact 0.4: The text contains the label "List of all compositions".

Retrieved items:
<item 1>
The Sleeping Beauty (ballet)
The work has become one of the classical repertoire 's most famous ballets . Contents ( hide ) 1 History 1.1 Composition history 1.2 Performance history 2 Roles
and original cast 3 Instrumentation 4 Roles 5 Synopsis 6 Structure 7 Versions by other hands 7.1 Piano arrangements 7.2 Aurora 's Wedding by Sergei Diaghilev 8
Trademark controversy 9 References 10 External links 10.1 Video samples 10.2 Scores 10.3 History History ( edit ) Composition history ( edit ) Tchaikovsky was
approached by the Director of the Imperial Theatres in St. Petersburg , Ivan Vsevolozhsky on 25 May 1888 about a possible ballet adaptation on the subject of
the story of Undine . It was later decided that Charles Perrault 's La Belle au bois dormant would be the story for which Tchaikovsky would compose the music
for the ballet . Tchaikovsky did not hesitate to accept the commission , although he was aware that his only previous ballet , Swan Lake , met with little
enthusiasm at that stage of his career .
</item 1>

<item 2>
The Sleeping Beauty (ballet)
This version has been performed by a number of central European ballet companies over the past two decades . Roles and original cast ( edit ) Role Mariinsky
1890 Bolshoi 1899 King Florestan Feliks Krzesiński ( Mathilde Kschessinska 's father ) Queen Giuseppina Cecchetti Princess Aurora , the Sleeping Beauty Carlotta
Brianza Lyubov Roslavleva Lilac Fairy Marie Petipa M. Grachevskaya Carabosse Enrico Cecchetti Vasiliy Geltser Prince Désiré Pavel Gerdt Ivan Khlyustin Bluebird
Enrico Cecchetti Princess Florine Varvara Nikitina The ballet 's premiere received more favorable accolades than Swan Lake from the press but Tchaikovsky never
had the luxury of being able to witness his work become an instant success in theatres outside of Russia . He died in 1893 .
</item 2>

<item 3>
The Sleeping Beauty (ballet)
1 in D major String Quartet No. 2 in F major String Quartet No. 3 in E ♭ minor Souvenir d'un lieu cher Piano Trio in A minor String Sextet in D minor ( Souvenir
de Florence ) Piano music Scherzo à la russe Souvenir de Hapsal The Seasons ( Les saisons ) Piano Sonata No.
</item 3>

<item 4>
The Sleeping Beauty (ballet)
The ballet scenario that Tchaikovsky worked on was based on the Brothers Grimm 's version of Perrault 's work entitled ' Dornröschen ' . In that version , the
Princess 's parents ( the King and the Queen ) survived the 100 - year sleep to celebrate the Princess 's wedding to the Prince . However , Vsevolozhsky
incorporated Perrault 's other characters from his stories into the ballet , such as Puss in Boots , Little Red Riding Hood , Cinderella , Bluebird , Bluebeard
, Ricky of the Tuft and Tom Thumb . Other French fairy tale characters to be featured are Beauty and the Beast , Pretty Goldilocks and The White Cat .
Regardless , Tchaikovsky was happy to inform the Director of the Imperial Theatre that he had great pleasure studying the work and had come away with adequate
inspiration to do it justice . The choreographer was Marius Petipa , ballet master of the Imperial Ballet , who wrote a very detailed list of instructions as to
the musical requirements .
</item 4>

<item 5>
The Sleeping Beauty (ballet)
2 in C ♯ minor Grand Piano Sonata in G major Portrayals Tchaikovsky ( 1969 film ) Music competitions International Tchaikovsky Competition International
Tchaikovsky Competition for Young Musicians Related articles Antonina Miliukova Belyayev circle Death Modest Ilyich Tchaikovsky Musical style Nadezhda von Meck
Pyotr Ilyich Tchaikovsky in popular media Symphonies by Pyotr Ilyich Tchaikovsky Tchaikovsky State House - Museum The Five Vladimir Davydov Book : Pyotr Ilyich
Tchaikovsky List of compositions by Pyotr Ilyich Tchaikovsky Category : Pyotr Ilyich Tchaikovsky `` Sleeping Beauty '' / `` Little Briar Rose '' by Charles
Perrault and the Brothers Grimm Theatre La Belle au Bois Dormant ( opera ) The Sleeping Beauty ( ballet ) Variants Sun , Moon ,
</item 5>

<item 6>
The Sleeping Beauty (ballet)
The Sleeping Beauty ( ballet ) - wikipedia The Sleeping Beauty ( ballet ) Jump to : navigation , search Ballets by Pyotr Ilyich Tchaikovsky Swan Lake ( 1876 )
Sleeping Beauty ( 1889 ) The Nutcracker ( 1892 ) List of all compositions The Sleeping Beauty ( Russian : Спящая красавица / Spyashchaya krasavitsa ) is a
ballet in a prologue and three acts , first performed in 1890 . The music was composed by Pyotr Ilyich Tchaikovsky ( his opus 66 ) . The score was completed in
1889 , and is the second of his three ballets . The original scenario was conceived by Ivan Vsevolozhsky , and is based on Charles Perrault 's La Belle au bois
dormant . The choreographer of the original production was Marius Petipa . The premiere performance took place at the Mariinsky Theatre in St. Petersburg on
January 15 , 1890 . The work has become one of the classical repertoire 's most famous ballets .
</item 6>

<item 7>
The Sleeping Beauty (ballet)
The Oprichnik Vakula the Smith Eugene Onegin The Maid of Orleans Mazepa ( or Mazeppa ) Cherevichki The Enchantress The Queen of Spades Iolanta Ballets Swan Lake
The Sleeping Beauty The Nutcracker Symphonies No. 1 in G minor ( `` Winter Daydreams '' ) No. 2 in C minor ( `` Little Russian '' ) No. 3 in D major ( `` Polish
'' ) No. 4 in F minor Symphony in B minor ( `` Manfred '' ) No. 5 in E minor No. 6 in B minor ( `` Pathétique '' ) No. 7 in E ♭ major ( unfinished ) Concertante
Piano Concerto No. 1 in B ♭ minor Piano Concerto No. 2 in G major Piano Concerto No.
</item 7>

<item 8>
The Sleeping Beauty (ballet)
With subtitles in Esperanto . Solo of Prince Désiré from Pas de deux ( Act 3 ) danced by Rudolf Nureyev ( from YouTube ) Scores ( edit ) Sleeping Beauty :
Scores at the International Music Score Library Project ( IMSLP ) History ( edit ) Tchaikovsky Research Ballets by Pyotr Ilyich Tchaikovsky Swan Lake , Op. 20
Sleeping Beauty , Op. 66 The Nutcracker , Op. 71 List of compositions by Pyotr Ilyich Tchaikovsky The ballets and * revivals of Marius Petipa in Russia 1847 --
59 Paquita ( * 1847 , * 1881 ) Le Diable amoureux ( as `` Satanella '' ) ( * 1848 ) Leda , the Swiss Milkmaid ( 1849 ) Giselle ( * 1850 , * 1884 , * 1899 , *
1903 ) The Star of Granada ( 1855 ) The Rose , the Violet , and the Butterfly ( 1857 ) Le Corsaire ( * 1858 , * 1863 , * 1868 ,
</item 8>

<item 9>
The Sleeping Beauty (ballet)
19 , as no. 18 was cut in the original production No. 18 Entr'acte symphonique ( solo for violin with light orchestral accompaniment , composed for Leopold Auer
, cut from the original production ) Scene II -- Le château de la belle au bois dormant No. 19 Scène du château de sommeil No. 20 Scène et final -- Le réveil
d'Aurore The Bluebird and Princess Florina ( Valeri Panov and Natalia Makarova ) from the 1964 Russian motion picture featuring artists of the Kirov Ballet .
Act III -- Les Noces de Désiré et d'Aurore No. 21 Marche No. 22 Grand polonaise dansée ( a.k.a. The Procession of the Fairy Tales ) Grand divertissement -- No.
23 Pas de quatre a . Entrée b . Variation de la fée - Or ( transferred by Petipa to Act II as a variation for Carlotta Brianza in the original production ) c .
</item 9>

<item 10>
The Sleeping Beauty (ballet)
* 1895 ) The Sleeping Beauty ( 1890 ) Nénuphar ( 1890 ) Kalkabrino ( 1891 ) A Fairy Tale ( 1891 ) La Sylphide ( * 1892 ) The Nutcracker ( 1892 ) Cinderella (
1893 ) The Awakening of Flora ( 1894 ) Swan Lake ( * 1895 ) The Little Humpbacked Horse ( as `` The Tsar Maiden '' ) ( * 1895 ) The Calvary Halt ( 1896 ) The
Pearl ( 1896 ) Bluebeard ( 1896 ) Raymonda ( 1898 ) 1900 -- 03 Les Ruses d'Amour or The Trial of Damis ( 1900 ) The Seasons ( 1900 ) Harlequinade ( 1900 ) The
Heart of the Marquis ( 1902 ) The Magic Mirror ( 1903 ) The Romance of the Rosebud and the Butterfly ( never presented ) Pyotr Ilyich Tchaikovsky Operas The
Voyevoda Undina The Oprichnik Vakula the Smith Eugene Onegin The Maid of Orleans Mazepa
</item 10>
=== END ITEM id=1c27f151661c0b0d ===

=== ITEM id=236ce53c81f797b9 ===
Gold sentences and facts:
Sentence 0: Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st Emmanuel Amunike Sporting CP 48 2nd George Weah Paris Saint - Germain 44 Rashidi Yekini Olympiacos 1995 1st George Weah Milan -- 2nd Emmanuel Amunike Sporting CP -- 3rd Daniel Amokachi Everton -- 1st Nwankwo Kanu Internazionale -- 2nd George Weah Milan -- 3rd Daniel Amokachi Beşiktaş -- 1st Victor Ikpeba Monaco 56 2nd Japhet N'Doram Monaco 40 3rd Taribo West Internazionale 35 1998 1st Mustapha Hadji Deportivo La Coruña 76 2nd Jay - Jay Okocha Paris Saint - Germain 74 3rd Sunday Oliseh Ajax 58 1999 1st Nwankwo Kanu Arsenal 46 2nd Samuel Kuffour Bayern Munich 44 3rd Ibrahima Bakayoko Marseille 42 2000 1st Patrick M'Boma Parma 123 2nd Lauren Mallorca 36 3rd Samuel Eto'o Mallorca 29 1st El Hadji Diouf Lens 93 2nd Samuel Kuffour Bayern Munich 66 3rd Samuel Eto'o Mallorca 34 2002 1st El Hadji Diouf Liverpool 93 2nd Papa Bouba Diop Lens 46 3rd Mido Ajax 42 2003 1st Samuel Eto'o Mallorca -- 2nd Didier Drogba Marseille -- 3rd Jay - Jay Okocha Bolton Wanderers -- 1st Samuel Eto'o Barcelona 116 2nd Didier Drogba Chelsea 90 3rd Jay - Jay Okocha Bolton Wanderers 68 2005 1st Samuel Eto'o Barcelona 108 2nd Didier Drogba Chelsea 106 3rd Michael Essien Chelsea 50 2006 1st Didier Drogba Chelsea 79 2nd Samuel Eto'o Barcelona 76 3rd Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74 2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea 43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311
  Fact 0.0: In 1992, Abedi Pele of Marseille ranked 1st with no points listed.
  Fact 0.1: In 1993, Rashidi Yekini of Vitória de Setúbal ranked 1st with no points listed.
  Fact 0.2: In 1994, Emmanuel Amunike of Sporting CP ranked 1st with 48 points.
  Fact 0.3: In 1994, George Weah of Paris Saint-Germain ranked 2nd with 44 points.
  Fact 0.4: In 1994, Rashidi Yekini of Olympiacos is listed with no rank shown.
  Fact 0.5: In 1995, George Weah of Milan ranked 1st with no points listed.
  Fact 0.6: In 1995, Emmanuel Amunike of Sporting CP ranked 2nd with no points listed.
  Fact 0.7: In 1995, Daniel Amokachi of Everton ranked 3rd with no points listed.
  Fact 0.8: In an entry with no year label, Nwankwo Kanu of Internazionale ranked 1st with no points listed.
  Fact 0.9: In an entry with no year label, George Weah of Milan ranked 2nd with no points listed.
  Fact 0.10: In an entry with no year label, Daniel Amokachi of Beşiktaş ranked 3rd with no points listed.
  Fact 0.11: In an entry with no year label, Victor Ikpeba of Monaco ranked 1st with 56 points.
  Fact 0.12: In an entry with no year label, Japhet N'Doram of Monaco ranked 2nd with 40 points.
  Fact 0.13: In an entry with no year label, Taribo West of Internazionale ranked 3rd with 35 points.
  Fact 0.14: In 1998, Mustapha Hadji of Deportivo La Coruña ranked 1st with 76 points.
  Fact 0.15: In 1998, Jay-Jay Okocha of Paris Saint-Germain ranked 2nd with 74 points.
  Fact 0.16: In 1998, Sunday Oliseh of Ajax ranked 3rd with 58 points.
  Fact 0.17: In 1999, Nwankwo Kanu of Arsenal ranked 1st with 46 points.
  Fact 0.18: In 1999, Samuel Kuffour of Bayern Munich ranked 2nd with 44 points.
  Fact 0.19: In 1999, Ibrahima Bakayoko of Marseille ranked 3rd with 42 points.
  Fact 0.20: In 2000, Patrick M'Boma of Parma ranked 1st with 123 points.
  Fact 0.21: In 2000, Lauren of Mallorca ranked 2nd with 36 points.
  Fact 0.22: In 2000, Samuel Eto'o of Mallorca ranked 3rd with 29 points.
  Fact 0.23: In an entry with no year label, El Hadji Diouf of Lens ranked 1st with 93 points.
  Fact 0.24: In an entry with no year label, Samuel Kuffour of Bayern Munich ranked 2nd with 66 points.
  Fact 0.25: In an entry with no year label, Samuel Eto'o of Mallorca ranked 3rd with 34 points.
  Fact 0.26: In 2002, El Hadji Diouf of Liverpool ranked 1st with 93 points.
  Fact 0.27: In 2002, Papa Bouba Diop of Lens ranked 2nd with 46 points.
  Fact 0.28: In 2002, Mido of Ajax ranked 3rd with 42 points.
  Fact 0.29: In 2003, Samuel Eto'o of Mallorca ranked 1st with no points listed.
  Fact 0.30: In 2003, Didier Drogba of Marseille ranked 2nd with no points listed.
  Fact 0.31: In 2003, Jay-Jay Okocha of Bolton Wanderers ranked 3rd with no points listed.
  Fact 0.32: In an entry with no year label, Samuel Eto'o of Barcelona ranked 1st with 116 points.
  Fact 0.33: In an entry with no year label, Didier Drogba of Chelsea ranked 2nd with 90 points.
  Fact 0.34: In an entry with no year label, Jay-Jay Okocha of Bolton Wanderers ranked 3rd with 68 points.
  Fact 0.35: In 2005, Samuel Eto'o of Barcelona ranked 1st with 108 points.
  Fact 0.36: In 2005, Didier Drogba of Chelsea ranked 2nd with 106 points.
  Fact 0.37: In 2005, Michael Essien of Chelsea ranked 3rd with 50 points.
  Fact 0.38: In 2006, Didier Drogba of Chelsea ranked 1st with 79 points.
  Fact 0.39: In 2006, Samuel Eto'o of Barcelona ranked 2nd with 76 points.
  Fact 0.40: In 2006, Michael Essien of Chelsea ranked 3rd with 36 points.
  Fact 0.41: In 2007, Frédéric Kanouté of Sevilla ranked 1st with no points listed.
  Fact 0.42: In 2007, Michael Essien of Chelsea ranked 2nd with no points listed.
  Fact 0.43: In 2007, Didier Drogba of Chelsea ranked 3rd with no points listed.
  Fact 0.44: In 2008, Emmanuel Adebayor of Arsenal ranked 1st with 74 points.
  Fact 0.45: In 2008, Mohamed Aboutrika of Al-Ahly ranked 2nd with 53 points.
  Fact 0.46: In 2008, Michael Essien of Chelsea ranked 3rd with 36 points.
  Fact 0.47: In 2009, Didier Drogba of Chelsea ranked 1st with 92 points.
  Fact 0.48: In 2009, Samuel Eto'o of Internazionale ranked 2nd with 69 points.
  Fact 0.49: In 2009, Michael Essien of Chelsea ranked 3rd with 43 points.
  Fact 0.50: In an entry with no year label, Samuel Eto'o of Internazionale ranked 1st with no points listed.
  Fact 0.51: In an entry with no year label, Asamoah Gyan of Sunderland ranked 2nd with no points listed.
  Fact 0.52: In an entry with no year label, Didier Drogba of Chelsea ranked 3rd with no points listed.
  Fact 0.53: In 2011, Yaya Touré of Manchester City ranked 1st with no points listed.
  Fact 0.54: In 2011, Seydou Keita of Barcelona ranked 2nd with no points listed.
  Fact 0.55: In 2011, André Ayew of Marseille ranked 3rd with no points listed.
  Fact 0.56: In 2012, Yaya Touré of Manchester City ranked 1st with no points listed.
  Fact 0.57: In 2012, Didier Drogba of Shanghai Shenhua ranked 2nd with no points listed.
  Fact 0.58: In 2012, Alex Song of Barcelona ranked 3rd with no points listed.
  Fact 0.59: In 2013, Yaya Touré of Manchester City ranked 1st with 373 points.
  Fact 0.60: In 2013, John Obi Mikel of Chelsea ranked 2nd with 265 points.
  Fact 0.61: In 2013, Didier Drogba of Galatasaray ranked 3rd with 236 points.
  Fact 0.62: In 2014, Yaya Touré of Manchester City ranked 1st with 175 points.
  Fact 0.63: In 2014, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 2nd with 120 points.
  Fact 0.64: In 2014, Vincent Enyeama of Lille ranked 3rd with 105 points.
  Fact 0.65: In 2015, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 1st with 143 points.
  Fact 0.66: In 2015, Yaya Touré of Manchester City ranked 2nd with 136 points.
  Fact 0.67: In 2015, André Ayew of Swansea City ranked 3rd with 112 points.
  Fact 0.68: In 2016, Riyad Mahrez of Leicester City ranked 1st with 361 points.
  Fact 0.69: In 2016, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 2nd with 313 points.
  Fact 0.70: In 2016, Sadio Mané of Liverpool ranked 3rd with 186 points.
  Fact 0.71: In 2017, Mohamed Salah of Liverpool ranked 1st with 625 points.
  Fact 0.72: In 2017, Sadio Mané of Liverpool ranked 2nd with 507 points.
  Fact 0.73: In 2017, Pierre-Emerick Aubameyang of Borussia Dortmund ranked 3rd with 311 points.

Retrieved items:
<item 1>
African Footballer of the Year
African Footballer of the Year - wikipedia African Footballer of the Year Jump to : navigation , search The African Footballer of the Year award , presented to
the best African footballer each year , has been conferred by the Confederation of African Football ( CAF ) since 1992 . An earlier African Footballer of the
Year Golden Ball award was given out between 1970 and 1994 by France Football magazine . The changes resulted in parallel Golden Ball awards given out to Abedi
Pele and George Weah in 1993 and 1994 by the magazine although the CAF sponsored awards for those years were won respectively by Rashidi Yekini and Emmanuel
Amuneke , as well as two awards given to Abedi Pelé in 1992 . France Football discontinued the election from 1995 after the European Ballon d'Or -- also awarded
by the magazine -- had been opened to all players in the European leagues . On 2 February 2008 , Sevilla 's Frédéric Kanouté was named the 2007 African
Footballer of the Year .
</item 1>

<item 2>
African Footballer of the Year
Keïta 1971 : Sunday 1972 : Souleymane 1973 : Bwanga 1974 : Moukila 1975 : Faras 1976 : Milla 1977 : Dhiab 1978 : Abdul Razak 1979 : N'Kono 1980 : Manga -
Onguéné 1981 : Belloumi 1982 : N'Kono 1983 : El Khatib 1984 : Abega 1985 : Timoumi 1986 : Zaki 1987 : Madjer 1988 : Bwalya 1989 : Weah 1990 : Milla 1991 : Abedi
Pele 1992 : Abedi Pele 1993 : Abedi Pele 1994 : Weah CAF award 1992 : Abedi Pele 1993 : Yekini 1994 : Amunike 1995 : Weah 1996 : Kanu 1997 : Ikpeba 1998 : Hadji
1999 : Kanu 2000 : M'Boma 2001 : Diouf 2002 :
</item 2>

<item 3>
African Footballer of the Year
Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd Roger Milla Tonnerre Yaoundé 24 Mamadou Aliou Kéïta Hafia 1976 1st Roger Milla Tonnerre Yaoundé 33 2nd Papa Camara
Hafia 32 3rd Ali Bencheikh MC Alger 27 1977 1st Tarak Dhiab Espérance Tunis 45 2nd Papa Camara Hafia 33 3rd Segun Odegbami Shooting Stars 29 1978 1st Karim
Abdul Razak Asante Kotoko 58 2nd Ali Bencheikh MC Alger 33 3rd Thomas N'Kono Canon Yaoundé 29 1979 1st Thomas N'Kono Canon Yaoundé 55 2nd Adolf Armah Hearts of
Oak 23 3rd Kerfalla Bangoura Horoya 15 1980 1st Jean Manga - Onguéné Canon Yaoundé 64 2nd Segun Odegbami Shooting Stars 41 3rd Théophile Abega Canon Yaoundé
</item 3>

<item 4>
African Footballer of the Year
It had already been replaced by an official award given out by the Confederation of African Football since 1992 . Year Rank Player Team Points 1970 1st Salif
Keïta Saint - Étienne 54 2nd Laurent Pokou ASEC Abidjan 28 Ali Abo Greisha Ismaily 1971 1st Ibrahim Sunday Asante Kotoko 29 2nd Robert Mensah Asante Kotoko 15
3rd François Ndoumbé Canon Yaoundé 13 1972 1st Chérif Souleymane Hafia 21 2nd Bwanga Tshimen TP Mazembe 16 3rd Petit Sory Hafia 14 1973 1st Bwanga Tshimen TP
Mazembe 49 2nd Kazadi Mwamba TP Mazembe 44 3rd Laurent Pokou ASEC Abidjan 41 1st Paul Moukila CARA Brazzaville 57 2nd Lobilo Boba Vita Club 32 3rd Hassan
Shehata Zamalek 28 1st Ahmed Faras Mohammédia 28 2nd
</item 4>

<item 5>
African Footballer of the Year
Michael Essien Chelsea 36 2007 1st Frédéric Kanouté Sevilla -- 2nd Michael Essien Chelsea -- 3rd Didier Drogba Chelsea -- 2008 1st Emmanuel Adebayor Arsenal 74
2nd Mohamed Aboutrika Al - Ahly 53 3rd Michael Essien Chelsea 36 2009 1st Didier Drogba Chelsea 92 2nd Samuel Eto'o Internazionale 69 3rd Michael Essien Chelsea
43 1st Samuel Eto'o Internazionale -- 2nd Asamoah Gyan Sunderland -- 3rd Didier Drogba Chelsea -- 2011 1st Yaya Touré Manchester City -- 2nd Seydou Keita
Barcelona -- 3rd André Ayew Marseille -- 2012 1st Yaya Touré Manchester City -- 2nd Didier Drogba Shanghai Shenhua -- 3rd Alex Song Barcelona -- 2013 1st Yaya
Touré Manchester City 373 2nd John Obi Mikel Chelsea 265 3rd Didier Drogba Galatasaray 236 2014
</item 5>

<item 6>
African Footballer of the Year
Cannes 52 1992 1st Abedi Pele Marseille 198 2nd George Weah Paris Saint - Germain 161 3rd Tony Yeboah Eintracht Frankfurt 64 1993 1st Abedi Pele Marseille 119
2nd Tony Yeboah Eintracht Frankfurt 117 3rd Rashidi Yekini Vitória de Setúbal 104 1994 1st George Weah Paris Saint - Germain 148 2nd Emmanuel Amunike Sporting
CP 133 3rd Daniel Amokachi Everton 99 CAF award ( 1992 -- present ) ( edit ) Samuel Eto'o won the award four times . Yaya Touré won the award four consecutive
times from 2011 to 2014 . Pierre - Emerick Aubameyang was the first player from Gabon to win the award in 2015 . Mohamed Salah is the latest player to win the
award in 2017 . Year Rank Player Team Points 1992 1st Abedi Pele Marseille -- 1993 1st Rashidi Yekini Vitória de Setúbal -- 1994 1st
</item 6>

<item 7>
African Footballer of the Year
Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2013 - 07 - 02 . Jump
up ^ `` African Player of the Year '' . rsssf.com. 2000 - 01 - 01 . Retrieved 2016 - 09 - 29 .
</item 7>

<item 8>
African Footballer of the Year
Club -- -- El Mokawloon -- -- Sunderland -- -- Girondins Bordeaux -- -- FC Mulhouse -- -- Hearts of Oak -- -- FC Sion -- -- Aydınspor -- -- JS Kabylie -- --
Bolton Wanderers -- -- Everton FC -- -- Ajax Amsterdam -- -- Ismaily SC -- -- Zamalek SC -- AS Cannes -- -- Stade Lavallois -- -- Lille OSC -- -- Stade Rennais
-- -- Olympiacos -- -- Horoya AC -- -- PSV Eindhoven -- -- OC Agaza -- -- Beşiktaş -- -- Galatasaray -- -- Swansea City -- -- See also ( edit ) BBC African
Footballer of the Year African Women Footballer of the Year FIFA World Player of the Year CAF Awards References ( edit ) Jump up ^ `` African Player of the Year
1993 '' . Rsssf.com. 2001 - 01 - 05 . Retrieved 2014 - 01 - 09 . Jump up ^ `` African Player of the Year 1994 '' . Rsssf.com. 2001 - 01 - 05 .
</item 8>

<item 9>
African Footballer of the Year
Chelsea 265 3rd Didier Drogba Galatasaray 236 2014 1st Yaya Touré Manchester City 175 2nd Pierre - Emerick Aubameyang Borussia Dortmund 120 3rd Vincent Enyeama
Lille 105 2015 1st Pierre - Emerick Aubameyang Borussia Dortmund 143 2nd Yaya Touré Manchester City 136 3rd André Ayew Swansea City 112 2016 1st Riyad Mahrez
Leicester City 361 2nd Pierre - Emerick Aubameyang Borussia Dortmund 313 3rd Sadio Mané Liverpool 186 2017 1st Mohamed Salah Liverpool 625 2nd Sadio Mané
Liverpool 507 3rd Pierre - Emerick Aubameyang Borussia Dortmund 311 Wins By Country ( edit ) Nation Winners Runners - up Third places Cameroon 11 8 11 Ivory
Coast 8 6 7 Ghana 6 7 7 Nigeria 5 6 7 Morocco 0 Algeria 5 Liberia -- Egypt Mali -- Senegal Guinea DR Congo -- Gabon Zambia -- Togo -- Tunisia -- -- Congo -- --
Chad -- -- Wins
</item 9>

<item 10>
African Footballer of the Year
Retrieved 2016 - 09 - 29 . Jump up ^ Kanoute named Africa 's best Jump up ^ Furious Drogba Slams CAF Over Kanoute Award And Pulls Out Football in Africa ( CAF )
National competitions Men Africa Cup of Nations African Nations Championship Africa Futsal Cup of Nations Africa Beach Soccer Cup of Nations Women Africa Women
Cup of Nations Women 's Olympic Qualifying Tournament African Games Youth competitions Men U-23 Cup of Nations U-20 Cup of Nations U-17 Cup of Nations African
Games Women U-20 Cup for Women U-17 Cup for Women Club competitions Current Champions League Confederation Cup Super Cup Defunct CAF Cup Cup Winners ' Cup Afro
- Asian Championship Lists and awards African Footballer of the Year African Women 's Footballer of the Year CAF Awards CAF 5 - Year Ranking List of African
national football team managers Sub-regions Northern Africa UNAF Western Africa WAFU Central Africa UNIFFAC Eastern Africa CECAFA Southern Africa COSAFA
CAFOnline.com African Footballer of the Year France Football award 1970 : Keïta 1971 : Sunday 1972 : Souleymane 1973 :
</item 10>
=== END ITEM id=236ce53c81f797b9 ===

=== ITEM id=c18e2a9914b46cf9 ===
Gold sentences and facts:
Sentence 0: The book became an instant hit , and the online version received about 20 million views as of 2009 ; many online readers requested a printed version . Kinney had agreed , and in April 2007 , Diary of a Wimpy Kid was published . To date , twelve Wimpy Kid books have been released , plus one do - it - yourself book and two movie diaries . In April 2009 , TIME named Kinney one of The World 's Most Influential People .
  Fact 0.0: The book became an instant hit.
  Fact 0.1: The online version of the book received about 20 million views as of 2009.
  Fact 0.2: Many online readers requested a printed version of the book.
  Fact 0.3: Kinney agreed to a printed version of the book.
  Fact 0.4: Diary of a Wimpy Kid was published in April 2007.
  Fact 0.5: To date, twelve Wimpy Kid books have been released.
  Fact 0.6: To date, one do-it-yourself Wimpy Kid book has been released.
  Fact 0.7: To date, two Wimpy Kid movie diaries have been released.
  Fact 0.8: In April 2009, TIME named Kinney one of The World's Most Influential People.
Sentence 1: # Title Date ISBN Diary of a Wimpy Kid April 1 , 2007 ISBN 978 - 0 - 8109 - 9313 - 6 The introduction to the series introduces the protagonist , Greg Heffley , his best friend Rowley , his middle school problems , Halloween , and the Cheese Touch . Rodrick Rules February 1 , 2008 ISBN 978 - 0 - 8109 - 9473 - 7 Greg must stop Rodrick from revealing the secrets that had happened last summer . The Last Straw January 13 , 2009 ISBN 978 - 0 - 8109 - 7068 - 7 When Greg 's father threatens to send Greg to a military academy , Greg must find a way to prevent this from happening . Dog Days October 12 , 2009 ISBN 978 - 0 - 8109 - 8391 - 5 Summer is here , but when the family has not that much money to go to the beach ( money is tight at that time ) , it might not be the best summer ever . 5 The Ugly Truth November 9 , 2010 ISBN 978 - 0 - 8109 - 8491 - 2 Greg is growing up , and he will have to face the challenges of this period ( like changing dentists , and a school lock - in ) . 6 Cabin Fever November 15 , 2011 ISBN 978 - 1 - 4197 - 0223 - 5 The family is stranded in the house as snow begins to fall . 7 The Third Wheel November 13 , 2012 ISBN 978 - 1 - 4197 - 0584 - 7 The Valentine 's Day dance is coming up , and Greg has no one go with . But when Michael Sampson ( Abigail 's date ) will not be going , things turn around . 8 Hard Luck November 5 , 2013 ISBN 978 - 1 - 4197 - 1132 - 9 Greg 's friend count is dropping as Rowley now has a girlfriend , named Abigail ( continuing the last book ) . 9 The Long Haul November 4 , 2014 ISBN 978 - 1 - 4197 - 1189 - 3 This is the family 's second summer trip , but instead , it 's a road trip . But with the Heffleys , you know what to expect from this disaster . 10 Old School November 3 , 2015 ISBN 978 - 1 - 4197 - 1701 - 7 The neighborhood is going to cut electricity for the weekend , and as the family 's car has a fender bender , it forces Greg to go to the farm trip . Will Greg survive the electricity - free days ? 11 Double Down November 1 , 2016 ISBN 978 - 1 - 4197 - 2344 - 5 Greg 's mom wants him to do something else than video games , so he makes a movie in collaboration with Rowley and joins the music program . 12 The Getaway November 7 , 2017 ISBN 978 - 1 - 4197 - 2545 - 6 Instead of celebrating the holidays at home , Greg and his family head on a tropical vacation to a resort . But what first starts as an relaxing trip , later turns into a Christmas nightmare . Will Greg be able to turn this vacation around ?
  Fact 1.0: Diary of a Wimpy Kid was released on April 1, 2007, with ISBN 978-0-8109-9313-6.
  Fact 1.1: The introduction to the Diary of a Wimpy Kid series introduces the protagonist Greg Heffley, his best friend Rowley, his middle school problems, Halloween, and the Cheese Touch.
  Fact 1.2: Rodrick Rules was released on February 1, 2008, with ISBN 978-0-8109-9473-7.
  Fact 1.3: In Rodrick Rules, Greg must stop Rodrick from revealing the secrets that happened last summer.
  Fact 1.4: The Last Straw was released on January 13, 2009, with ISBN 978-0-8109-7068-7.
  Fact 1.5: In The Last Straw, when Greg's father threatens to send Greg to a military academy, Greg must find a way to prevent this.
  Fact 1.6: Dog Days was released on October 12, 2009, with ISBN 978-0-8109-8391-5.
  Fact 1.7: In Dog Days, summer is here, but the family does not have much money to go to the beach, so it might not be the best summer ever.
  Fact 1.8: The Ugly Truth was released on November 9, 2010, with ISBN 978-0-8109-8491-2.
  Fact 1.9: In The Ugly Truth, Greg is growing up and will have to face the challenges of this period, like changing dentists and a school lock-in.
  Fact 1.10: Cabin Fever was released on November 15, 2011, with ISBN 978-1-4197-0223-5.
  Fact 1.11: In Cabin Fever, the family is stranded in the house as snow begins to fall.
  Fact 1.12: The Third Wheel was released on November 13, 2012, with ISBN 978-1-4197-0584-7.
  Fact 1.13: In The Third Wheel, the Valentine's Day dance is coming up and Greg has no one to go with.
  Fact 1.14: In The Third Wheel, things turn around when Michael Sampson, Abigail's date, will not be going.
  Fact 1.15: Hard Luck was released on November 5, 2013, with ISBN 978-1-4197-1132-9.
  Fact 1.16: In Hard Luck, Greg's friend count is dropping as Rowley now has a girlfriend, named Abigail.
  Fact 1.17: Hard Luck continues the story from The Third Wheel.
  Fact 1.18: The Long Haul was released on November 4, 2014, with ISBN 978-1-4197-1189-3.
  Fact 1.19: In The Long Haul, this is the family's second summer trip, and it is a road trip.
  Fact 1.20: Old School was released on November 3, 2015, with ISBN 978-1-4197-1701-7.
  Fact 1.21: In Old School, the neighborhood is going to cut electricity for the weekend.
  Fact 1.22: In Old School, the family's car has a fender bender, which forces Greg to go on the farm trip.
  Fact 1.23: Old School asks whether Greg will survive the electricity-free days.
  Fact 1.24: Double Down was released on November 1, 2016, with ISBN 978-1-4197-2344-5.
  Fact 1.25: In Double Down, Greg's mom wants him to do something other than video games, so he makes a movie in collaboration with Rowley.
  Fact 1.26: In Double Down, Greg joins the music program.
  Fact 1.27: The Getaway was released on November 7, 2017, with ISBN 978-1-4197-2545-6.
  Fact 1.28: In The Getaway, instead of celebrating the holidays at home, Greg and his family head on a tropical vacation to a resort.
  Fact 1.29: In The Getaway, what first starts as a relaxing trip later turns into a Christmas nightmare.
  Fact 1.30: The Getaway asks whether Greg will be able to turn the vacation around.
Sentence 2: Diary of a Wimpy Kid The logo of the series , appearing on the book covers , blogs , web comic , and the films . Diary of a Wimpy Kid Diary of a Wimpy Kid : Rodrick Rules Diary of a Wimpy Kid : The Last Straw Diary of a Wimpy Kid : Dog Days Diary of a Wimpy Kid : The Ugly Truth Diary of a Wimpy Kid : Cabin Fever Diary of a Wimpy Kid : The Third Wheel Diary of a Wimpy Kid : Hard Luck Diary of a Wimpy Kid : The Long Haul Diary of a Wimpy Kid : Old School Diary of a Wimpy Kid : Double Down Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Genre Realistic fiction Publisher Scholastic Publishing / Amulet Books / Puffin Books Published April 1 , 2007 -- present Media type Print ( hardback & paperback ) Audiobook E-book
  Fact 2.0: The logo of the Diary of a Wimpy Kid series appears on the book covers, blogs, web comic, and films.
  Fact 2.1: The Diary of a Wimpy Kid series includes the books Rodrick Rules, The Last Straw, Dog Days, The Ugly Truth, Cabin Fever, The Third Wheel, Hard Luck, The Long Haul, Old School, Double Down, and The Getaway.
  Fact 2.2: The Diary of a Wimpy Kid series was written by Jeff Kinney.
  Fact 2.3: The Diary of a Wimpy Kid series is from the United States.
  Fact 2.4: The Diary of a Wimpy Kid series is in English.
  Fact 2.5: The Diary of a Wimpy Kid series is in the realistic fiction genre.
  Fact 2.6: The Diary of a Wimpy Kid series is published by Scholastic Publishing, Amulet Books, and Puffin Books.
  Fact 2.7: The Diary of a Wimpy Kid series was published from April 1, 2007 to the present.
  Fact 2.8: The Diary of a Wimpy Kid series is available in print (hardback and paperback), as an audiobook, and as an e-book.

Retrieved items:
<item 1>
Diary of a Wimpy Kid (book series)
Contents ( hide ) 1 Background 2 Chronology 2.1 Supplementary books 2.2 Toys and games 2.3 Audiobooks 3 Characters 4 Achievements 4.1 Awards and honors 4.2
Commercial success 5 Films 6 Musical 7 References 8 External links Background ( edit ) In January 1998 , Diary of a Wimpy Kid officially began , when Kinney
brought up an idea about a middle - school weakling named Greg Heffley , who wrote about his personal life . In May 2004 , FunBrain and Kinney released an
online version of Diary of a Wimpy Kid . The website made daily entries until June 2005 . He worked on his book for almost eight years , before showing it to a
publisher in New York . In February 2006 , during the New York Comic Con , Jeff signed a multi book deal with publisher Harry N. Abrams , Inc. , to turn Diary
of a Wimpy Kid into a print series .
</item 1>

<item 2>
Diary of a Wimpy Kid (book series)
ISBN 0 - 8109 - 9313 - 9 . ^ Jump up to : `` '' Diary of a Wimpy Kid ( 2010 ) '' on `` . Comingsoon.net . March 19 , 2010 . Retrieved May 7 , 2011 . ^ Jump up
to : `` Diary of a Wimpy Kid '' . FunBrain . Family Education Network . Retrieved March 5 , 2011 . Jump up ^ `` WIMPY 2 LAST '' . 3 November 2015 . Retrieved 6
March 2016 . Jump up ^ The 2009 TIME 100 : Jeff Kinney Time magazine 's 2009 list of The World 's Most Influential People Jump up ^ Kinney , Jeff . `` The Wimpy
Kid Movie Diary : The Next Chapter '' . Retrieved March 28 , 2017 . Jump up ^ Nickelodeon Kids ' Choice Awards , Nick.com Jump up ^ Dixler , Elsa . `` The New
York Times Best Sellers : Children 's Books '' . The New York Times .
</item 2>

<item 3>
Diary of a Wimpy Kid (book series)
USA Today , which ranks best sellers based on sales alone regardless of genre or intended audience , has listed Diary of a Wimpy Kid in its top 150 , peaking at
number 8 , for 141 weeks as of April 4 , 2010 . As of the same date , Rodrick Rules has been on the list for 117 weeks ( peaking at number 4 ) , The Last Straw
has been on the list for 65 weeks ( peaking at number 1 ) , Dog Days has been on the list for all 25 weeks of its publication ( peaking at number 1 ) , and The
Wimpy Kid Movie Diary has been listed for all three weeks of its publication , peaking at number 2 . The series has sold over 194 million books in more than 30
countries . The series has also generated over $500 million in revenue . Films ( edit ) Main article : Diary of a Wimpy Kid ( film series ) This section needs
to be updated . Please update this article to reflect recent events or newly available information .
</item 3>

<item 4>
Diary of a Wimpy Kid (book series)
2007 ) Diary of a Wimpy Kid : Rodrick Rules ( February 1 , 2008 ) Diary of a Wimpy Kid : The Last Straw ( January 13 , 2009 ) Diary of a Wimpy Kid : Dog Days (
October 12 , 2009 ) Diary of a Wimpy Kid : The Ugly Truth ( November 9 , 2010 ) Diary of a Wimpy Kid : Cabin Fever ( November 15 , 2011 ) Diary of a Wimpy Kid :
The Third Wheel ( November 13 , 2012 ) Diary of a Wimpy Kid : Hard Luck ( November 5 , 2013 ) Diary of a Wimpy Kid : The Long Haul ( November 4 , 2014 ) Diary
of a Wimpy Kid : Old School ( November 3 , 2015 ) Diary of a Wimpy Kid : Double Down ( November 1 , 2016 ) Diary of a Wimpy Kid : The Getaway ( November 7 ,
</item 4>

<item 5>
Diary of a Wimpy Kid (book series)
7 The Third Wheel November 13 , 2012 ISBN 978 - 1 - 4197 - 0584 - 7 The Valentine 's Day dance is coming up , and Greg has no one go with . But when Michael
Sampson ( Abigail 's date ) will not be going , things turn around . 8 Hard Luck November 5 , 2013 ISBN 978 - 1 - 4197 - 1132 - 9 Greg 's friend count is
dropping as Rowley now has a girlfriend , named Abigail ( continuing the last book ) . 9 The Long Haul November 4 , 2014 ISBN 978 - 1 - 4197 - 1189 - 3 This is
the family 's second summer trip , but instead , it 's a road trip . But with the Heffleys , you know what to expect from this disaster .
</item 5>

<item 6>
Diary of a Wimpy Kid (book series)
Diary of a Wimpy Kid Diary of a Wimpy Kid : Rodrick Rules Diary of a Wimpy Kid : The Last Straw Diary of a Wimpy Kid : Dog Days Diary of a Wimpy Kid : The Ugly
Truth Diary of a Wimpy Kid : Cabin Fever Diary of a Wimpy Kid : The Third Wheel Diary of a Wimpy Kid : Hard Luck Diary of a Wimpy Kid : The Long Haul Diary of a
Wimpy Kid : Old School Diary of a Wimpy Kid : Double Down Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Genre
Realistic fiction Publisher Scholastic Publishing / Amulet Books / Puffin Books Published April 1 , 2007 -- present Media type Print ( hardback & paperback )
Audiobook E-book Diary of a Wimpy Kid is a series of fiction books written by the American author and cartoonist Jeff Kinney . All the main books are the
journals of the main character , Greg Heffley .
</item 6>

<item 7>
Diary of a Wimpy Kid (book series)
All the main books are the journals of the main character , Greg Heffley . Befitting a teen 's diary , the books are filled with hand - written notes and simple
drawings of Greg 's daily adventures . Since the release of the online version in May 2004 , most of the books have garnered positive reviews and commercial
success . As of February 2008 , it has been purchased one million times . The first , second , fourth , and ninth installments have been adapted into films by
20th Century Fox . The website Poptropica released islands in the theme of Diary of a Wimpy Kid called Wimpy Wonderland and Wimpy Boardwalk , both of which
credit Kinney in an ambiguous contributor 's role .
</item 7>

<item 8>
Diary of a Wimpy Kid (book series)
Will Greg be able to turn this vacation around ? Supplementary books ( edit ) Diary of a Wimpy Kid Do - It - Yourself Book ( published between Rodrick Rules and
The Last Straw ) is an activity book that features a blank section encouraging readers to keep their own journal in the style of the books . The Wimpy Kid Do -
It - Yourself Book : Extended Edition ( published after Wimpy Kid Movie Diary 2 and before Cabin Fever ) is the same book as the first Do - It - Yourself Book ,
but with extra 60 pages and 16 more full - color comics . The Wimpy Kid Movie Diary is a book about the making of the first film , which features stills and
brand - new illustrations . Jeff Kinney dedicated the book to the film 's main stars , Zachary Gordon and Robert Capron . The Wimpy Kid Movie Diary 2 : Rodrick
Rules is an updated version of the first book , now featuring stills from the second film Rodrick Rules , more new illustrations , and information on how the
second film was made .
</item 8>

<item 9>
Diary of a Wimpy Kid (book series)
Diary of a Wimpy Kid ( book series ) - Wikipedia Diary of a Wimpy Kid ( book series ) This is the latest accepted revision , reviewed on 22 April 2018 . Jump to
: navigation , search `` Diary of a Wimpy Kid ( series ) '' redirects here . For the film series , see Diary of a Wimpy Kid ( film series ) . Diary of a Wimpy
Kid The logo of the series , appearing on the book covers , blogs , web comic , and the films .
</item 9>

<item 10>
Diary of a Wimpy Kid (book series)
, to turn Diary of a Wimpy Kid into a print series . The book became an instant hit , and the online version received about 20 million views as of 2009 ; many
online readers requested a printed version . Kinney had agreed , and in April 2007 , Diary of a Wimpy Kid was published . To date , twelve Wimpy Kid books have
been released , plus one do - it - yourself book and two movie diaries . In April 2009 , TIME named Kinney one of The World 's Most Influential People .
Chronology ( edit ) # Title Date ISBN Diary of a Wimpy Kid April 1 , 2007 ISBN 978 - 0 - 8109 - 9313 - 6 The introduction to the series introduces the
protagonist , Greg Heffley , his best friend Rowley , his middle school problems , Halloween , and the Cheese Touch . Rodrick Rules February 1 , 2008 ISBN 978 -
0 - 8109 - 9473 - 7 Greg must stop Rodrick from revealing the secrets that had happened last summer .
</item 10>
=== END ITEM id=c18e2a9914b46cf9 ===

=== ITEM id=e8f1a02191228d4a ===
Gold sentences and facts:
Sentence 0: The Country Music Hall of Fame and Museum in Nashville , Tennessee , is one of the world 's largest museums and research centers dedicated to the preservation and interpretation of American vernacular music . Chartered in 1964 , the museum has amassed one of the world 's most extensive musical collections .
  Fact 0.0: The Country Music Hall of Fame and Museum is in Nashville, Tennessee.
  Fact 0.1: The Country Music Hall of Fame and Museum is one of the world's largest museums dedicated to the preservation and interpretation of American vernacular music.
  Fact 0.2: The Country Music Hall of Fame and Museum is one of the world's largest research centers dedicated to the preservation and interpretation of American vernacular music.
  Fact 0.3: The Country Music Hall of Fame and Museum was chartered in 1964.
  Fact 0.4: The Country Music Hall of Fame and Museum has amassed one of the world's most extensive musical collections.
Sentence 1: Country Music Hall of Fame and Museum Established 1961 ( 1961 ) Location 222 5th Ave S Nashville , Tennessee 37203 Type Hall of fame Visitors 1.1 million in 2015 Director Kyle Young Website www.countrymusichalloffame.org
  Fact 1.0: The Country Music Hall of Fame and Museum was established in 1961.
  Fact 1.1: The Country Music Hall of Fame and Museum is located at 222 5th Ave S, Nashville, Tennessee 37203.
  Fact 1.2: The type of the Country Music Hall of Fame and Museum is Hall of fame.
  Fact 1.3: The Country Music Hall of Fame and Museum had 1.1 million visitors in 2015.
  Fact 1.4: The director of the Country Music Hall of Fame and Museum is Kyle Young.
  Fact 1.5: The website of the Country Music Hall of Fame and Museum is www.countrymusichalloffame.org.

Retrieved items:
<item 1>
Country Music Hall of Fame and Museum
Award recipients are determined through a two - stage balloting process ; the first round of voting narrows each category to five candidates ; the second round
selects winners . The large select committee of electors that votes on Hall of Fame membership is composed of CMA members who have participated in the country
music industry for at least ten years . New Hall of Fame members receive special recognition in ceremonies at the Country Music Hall of Fame and Museum . Bas -
relief portraits cast in bronze honoring each Hall of Fame member were originally displayed at the Tennessee State Museum in downtown Nashville until the
Country Music Hall of Fame and Museum opened its own building in April 1967 ; in this barn - roofed facility at the head of Music Row , the bronze plaques
comprised a special exhibit . Today the plaques are displayed in a seventy - foot - high rotunda at the museum 's enlarged downtown Nashville facility . The
museum collection ( edit ) The museum 's archival and library storage , allows for the cultivation of the museum 's collection and includes more than 2.5
million artifacts .
</item 1>

<item 2>
Country Music Hall of Fame and Museum
Located at the head of Music Row , the museum was erected on the site of a small Nashville city park . At this point , artifacts began to be displayed and a
small library was begun in a loft above one of the museum 's galleries . Early in the 1970s the basement of the museum building was partially complete , and
library expansion began , embracing not only recordings but also books and periodicals , sheet music and songbooks , photographs , business documents , and
other materials . At the outset , CMA staff had run the museum , but by 1972 the museum ( already governed by its own independent board of directors ) acquired
its own small staff , which has steadily increased to over 150 full - time professionals . Building expansion took place in 1974 , 1977 , and 1984 to store and
display the museum 's growing collection of costumes , films , historic cars , musical instruments , and other artifacts . An education department was created
to conduct ongoing programs with Middle Tennessee schools , an oral history program was begun , and a publications department was launched to handle books , as
well as the Journal of Country Music .
</item 2>

<item 3>
Country Music Hall of Fame and Museum
Current museum ( edit ) To become more accessible , in May 2001 the Country Music Hall of Fame and Museum moved to a new , 130,000 - square - foot ( 12,000 m )
facility in the heart of downtown Nashville 's arts and entertainment district . In 2014 , the museum unveiled a $100 million expansion , doubling its size to
350,000 square feet of galleries , archival storage , education classrooms , retail stores , and special event space . In the museum 's core exhibition , Sing
Me Back Home : A Journey Through Country Music , visitors are immersed in the history and sounds of country music , its origins and traditions , and the stories
and voices of many of its architects . The story is revealed through artifacts , photographs , and text panels , recorded sound , vintage video , and
interactive touchscreens . Sing Me Back Home is enhanced by numerous , rotating limited - engagement exhibits . The new ACM Gallery and the Dinah and Fred
Gretsch Family Gallery offer visitors a hands - on immersion into today 's country music with artifacts from today 's country stars and a series of technology -
enhanced activities .
</item 3>

<item 4>
Country Music Hall of Fame and Museum
Country Music Hall of Fame and museum - wikipedia Country Music Hall of Fame and museum Jump to : navigation , search This article needs additional citations
for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2016 ) (
Learn how and when to remove this template message ) Country Music Hall of Fame and Museum Established 1961 ( 1961 ) Location 222 5th Ave S Nashville ,
Tennessee 37203 Type Hall of fame Visitors 1.1 million in 2015 Director Kyle Young Website www.countrymusichalloffame.org The Country Music Hall of Fame and
Museum in Nashville , Tennessee , is one of the world 's largest museums and research centers dedicated to the preservation and interpretation of American
vernacular music . Chartered in 1964 , the museum has amassed one of the world 's most extensive musical collections .
</item 4>

<item 5>
Country Music Hall of Fame and Museum
Nashville , Tenn. : Country Music Foundation , 1990 . N.B. : The structure as described in this document is the original one which had opened in 1967 ; the book
( amply ill . in b&w and col . ) describes the work , collections , etc. of the center 's Hall of Fame , Museum ( which includes a library ) , and recording
studio , and also gives an account of the development and various aspects of the country music industry . ISBN 0 - 915608 - 12 - X References ( edit ) ^ Jump up
to : `` About '' . Country Music Hall of Fame . Retrieved January 14 , 2016 . Jump up ^ McCall , Michael ( 2012 ) . The Encyclopedia of Country Music Second
Edition . Oxford University Press . pp. 105 -- 106 . ISBN 0195395638 . Jump up ^ `` Home '' . Studio B. Retrieved August 26 , 2017 . Jump up ^ `` Country Music
Hall of Fame promotes four to VP roles '' . Nashville Post .
</item 5>

<item 6>
Country Music Hall of Fame and Museum
Contents ( hide ) 1 History of the museum 2 Current museum 3 The Country Music Hall of Fame 4 The Museum collection 5 Architecture and design 6 Timeline 7
Senior leadership 8 See also 9 Further reading 10 References 11 External links History of the museum ( edit ) The Country Music Hall of Fame and Museum is one
of the world 's largest and most active popular music research centers and the world 's largest repository of country music artifacts . Early in the 1960s , as
the Country Music Association 's campaign to publicize country music was accelerating , CMA leaders determined that a new organization was needed to operate a
country music museum and to carry out research and education activities beyond CMA 's scope as a trade organization . Toward this end , the nonprofit Country
Music Foundation ( CMF ) was chartered by the state of Tennessee in 1964 to collect , preserve , and publicize information and artifacts relating to the history
of country music . Through CMF , industry leaders raised money to build the Country Music Hall of Fame and Museum , which opened on April 1 , 1967 .
</item 6>

<item 7>
Country Music Hall of Fame and Museum
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Country Music Hall of Fame and Museum
In addition to the galleries , the museum has the 776 - seat CMA Theater , the Taylor Swift Education Center , and multi-purpose event rental spaces . Other
historic properties of the Country Music Hall of Fame and Museum include the letterpress operation Hatch Show Print ( located inside the museum ) and Historic
RCA Studio B ( located on Music Row ) , Nashville 's oldest surviving recording studio , where recordings by Elvis Presley , Dolly Parton , Waylon Jennings ,
and many others were made . The Country Music Hall of Fame and Museum has developed multiple platforms to make its collection accessible to a wider audience .
From weekly instrument demonstrations to its flagship songwriting program for schools , Words & Music , the museum offers an aggressive schedule of educational
programs . The museum also operates CMF Records , a Grammy - winning re-issue label ( The Complete Hank Williams and Night Train to Nashville : Music City
Rhythm & Blues , 1945 - 1970 ) ; and CMF Press , a book publishing arm that releases exhibit - related books in cooperation with Vanderbilt University Press and
other major trade publishing houses .
</item 8>

<item 9>
Graph query row:
entity: Country Music Hall of Fame and Museum
location_path: ['Country Music Hall of Fame and Museum', 'Nashville', 'Tennessee']
</item 9>

<item 10>
Country Music Hall of Fame and Museum
Nashville Post . February 17 , 2015 . Retrieved January 19 , 2016 . External links ( edit ) Wikimedia Commons has media related to Country Music Hall of Fame
and Museum . Official website ( hide ) Nashville landmarks Existing AT&T Building Belmont Mansion Bicentennial Capitol Mall State Park Bluebird Café Bridgestone
Arena Centennial Park Cheekwood Country Music Hall of Fame First Tennessee Park Fort Nashborough Fort Negley Frist Center for the Visual Arts Gaylord Opryland
Resort Grand Ole Opry Greer Stadium The Hermitage Life & Casualty Tower Memorial Gymnasium Music City Center Nashville City Cemetery Nashville International
Airport Nashville Zoo at Grassmere Nissan Stadium Parthenon Ryman Auditorium Schermerhorn Symphony Center John Seigenthaler Pedestrian Bridge Tennessee
Performing Arts Center Tennessee State Capitol Tennessee State Museum Tennessee Tower Union Station Vanderbilt Stadium Former Opryland USA Starwood Amphitheatre
Sulphur Dell Coordinates : 36 ° 09 ′ 29.80 '' N 86 ° 46 ′ 34.18 '' W ﻿ / ﻿ 36.1582778 ° N 86.7761611 ° W ﻿ / 36.
</item 10>
=== END ITEM id=e8f1a02191228d4a ===

=== ITEM id=5679ee1cc71c461b ===
Gold sentences and facts:
Sentence 0: In 1910 , in conjunction with advocacy work led by Florence Kelley of the National Consumer League , the Women 's Trade Union League ( WTLU ) of Massachusetts under the leadership of Elizabeth Evans took up the cause of minimum wage legislation in Massachusetts . Over the next two years , a coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state . In 1912 , Massachusetts passed the first American minimum wage legislation , which established a state commission for recommending non-compulsory minimum wages for women and children . Within eight years , at least thirteen U.S. states and the District of Columbia would pass minimum wage laws , with pressure being placed on state legislatures by the National Consumers League in a coalition with other women 's voluntary associations and organized labor . The United States Supreme Court of the Lochner era consistently invalidated compulsory minimum wage laws . Advocates for these minimum wage laws hoped that they would be upheld under the precedent of Muller v. Oregon , which upheld maximum working hours laws for women on the grounds that women required special protection that men did not . However , the Court did not extend this principle to minimum wage laws , considering the latter as interfering with the ability of employers to freely negotiate wage contracts with employees .
  Fact 0.0: In 1910, the Women's Trade Union League (WTLU) of Massachusetts, under the leadership of Elizabeth Evans, took up the cause of minimum wage legislation in Massachusetts.
  Fact 0.1: The 1910 effort was in conjunction with advocacy work led by Florence Kelley of the National Consumer League.
  Fact 0.2: Over the next two years, a coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state.
  Fact 0.3: In 1912, Massachusetts passed the first American minimum wage legislation.
  Fact 0.4: The 1912 Massachusetts law established a state commission for recommending non-compulsory minimum wages for women and children.
  Fact 0.5: Within eight years, at least thirteen U.S. states and the District of Columbia would pass minimum wage laws.
  Fact 0.6: Pressure on state legislatures came from the National Consumers League, in a coalition with other women's voluntary associations and organized labor.
  Fact 0.7: The United States Supreme Court of the Lochner era consistently invalidated compulsory minimum wage laws.
  Fact 0.8: Advocates for minimum wage laws hoped they would be upheld under the precedent of Muller v. Oregon.
  Fact 0.9: Muller v. Oregon upheld maximum working hours laws for women on the grounds that women required special protection that men did not.
  Fact 0.10: The Court did not extend the Muller v. Oregon principle to minimum wage laws.
  Fact 0.11: The Court considered minimum wage laws as interfering with the ability of employers to freely negotiate wage contracts with employees.
Sentence 1: In 1933 , the Roosevelt administration made the first attempt at establishing a national minimum wage , when a $0.25 per hour standard was set as part of the National Industrial Recovery Act . However , in the 1935 court case Schechter Poultry Corp. v. United States ( 295 U.S. 495 ) , the US Supreme Court declared the act unconstitutional , and the minimum wage was abolished . In 1938 , the minimum wage was re-established pursuant to the Fair Labor Standards Act , once again at $0.25 per hour ( $4.78 in 2017 dollars ) . In 1941 , the Supreme Court upheld the Fair Labor Standards Act in United States v. Darby Lumber Co. , holding that Congress had the power under the Commerce Clause to regulate employment conditions .
  Fact 1.0: In 1933, the Roosevelt administration made the first attempt at establishing a national minimum wage.
  Fact 1.1: The 1933 minimum wage attempt set a $0.25 per hour standard as part of the National Industrial Recovery Act.
  Fact 1.2: In the 1935 court case Schechter Poultry Corp. v. United States (295 U.S. 495), the US Supreme Court declared the National Industrial Recovery Act unconstitutional.
  Fact 1.3: The national minimum wage was abolished after the Schechter Poultry ruling.
  Fact 1.4: In 1938, the minimum wage was re-established under the Fair Labor Standards Act.
  Fact 1.5: The 1938 minimum wage was $0.25 per hour.
  Fact 1.6: The 1938 rate of $0.25 per hour equals $4.78 in 2017 dollars.
  Fact 1.7: In 1941, the Supreme Court upheld the Fair Labor Standards Act in United States v. Darby Lumber Co.
  Fact 1.8: In United States v. Darby Lumber Co., the Court held that Congress had the power under the Commerce Clause to regulate employment conditions.

Retrieved items:
<item 1>
Minimum wage in the United States
The Annals of the American Academy of Political and Social Science . 48 : 27 . Jump up ^ Broda , Rudolf ; United States . Bureau of Labor Statistics ( December
1928 ) . Minimum Wage Legislation in Various Countries : Bulletin of the United States Bureau of Labor Statistics , No. 467 ( PDF ) . Bulletin of the United
States Bureau of Labor Statistics . G.P.O. p. 11 . Jump up ^ Jerold L. , , Waltman ( 2000 ) . The politics of the minimum wage . Urbana : University of Illinois
Press . p. 11 . ISBN 9780252025457 . OCLC 42072067 . Jump up ^ Vivien. , Hart , ( 2001 ) . Bound by Our Constitution : Women , Workers , and the Minimum Wage .
Princeton : Princeton University Press . p. 63 . ISBN 9781400821563 . OCLC 700688619 . Jump up ^ Vivien. , Hart , ( 2001 ) .
</item 1>

<item 2>
Minimum wage in the United States
Minimum wage in the United States - wikipedia Minimum wage in the United States Jump to : navigation , search Minimum wage by U.S. state and territory , as of
January 1 , 2018 . Versus the federal rate : Higher Same Lower No state minimum wage law The minimum wage in the United States is set by US labor law and a
range of state and local laws . Employers generally have to pay workers the highest minimum wage prescribed by federal , state , and local law . Since July 24 ,
2009 , the federal government has mandated a nationwide minimum wage of $7.25 per hour . As of January 2018 , there were 29 states with a minimum wage higher
than the federal minimum . From 2017 to 2018 , eight states increased their minimum wage levels through automatic adjustments , while increases in eleven other
states occurred through referendum or legislative action . Washington has the highest state minimum wage at $11.50 per hour . Using 2017 inflation - adjusted
dollars , the federal minimum wage peaked at $11.53 per hour in 1968 .
</item 2>

<item 3>
Minimum wage in the United States
From January 1981 to April 1990 , the minimum wage was frozen at $3.35 per hour , then a record - setting minimum wage freeze . From September 1 , 1997 through
July 23 , 2007 , the federal minimum wage remained constant at $5.15 per hour , breaking the old record . From the United States Department of Labor .
Employment Standards Administration . Wage and Hour Division , the source page has a clickable US map with current and projected state - by - state minimum wage
rates for each state . Some government entities , such as counties and cities , observe minimum wages that are higher than the state as a whole . One notable
example of this is Santa Fe , New Mexico , whose $9.50 per hour minimum wage was the highest in the nation , until San Francisco increased its minimum wage to
$9.79 in 2009 . Another device to increase wages , living wage ordinances , generally apply only to businesses that are under contract to the local government
itself . Since 1984 , the purchasing power of the federal minimum wage has decreased .
</item 3>

<item 4>
Minimum wage in the United States
Bureau of Labor Statistics '' . Bls.gov . Retrieved 18 October 2017 . Jump up ^ Tritch , Teresa ( March 7 , 2014 ) . `` F.D.R. Makes the Case for the Minimum
Wage '' . New York Times . Retrieved March 7 , 2014 . Jump up ^ Broda , Rudolf ; United States . Bureau of Labor Statistics ( December 1928 ) . Minimum Wage
Legislation in Various Countries : Bulletin of the United States Bureau of Labor Statistics , No. 467 ( PDF ) . Bulletin of the United States Bureau of Labor
Statistics . G.P.O. p. 8 . Jump up ^ Willis J. , Nordlund ( 1997 ) . The quest for a living wage : the history of the federal minimum wage program . Westport ,
Conn. : Greenwood Press . p. 2 . ISBN 9780313264122 . OCLC 33983425 . Jump up ^ Hammond , Matthew B. ( 1913 ) . `` The Minimum Wage in Great Britain and
Australia '' .
</item 4>

<item 5>
Minimum wage in the United States
, holding that Congress had the power under the Commerce Clause to regulate employment conditions . The 1938 minimum wage law only applied to `` employees
engaged in interstate commerce or in the production of goods for interstate commerce , '' but in amendments in 1961 and 1966 , the federal minimum wage was
extended ( with slightly different rates ) to employees in large retail and service enterprises , local transportation and construction , state and local
government employees , as well as other smaller expansions ; a grandfather clause in 1990 drew most employees into the purview of federal minimum wage policy ,
which now set the wage at $3.80 . 21st century ( edit ) Legislation ( edit ) Play media An opinion video from the Economic Policy Institute about working for
the tipped minimum wage of $2.13 per hour . In 2006 , voters in six states ( Arizona , Colorado , Missouri , Montana , Nevada , and Ohio ) approved statewide
increases in the state minimum wage . The amounts of these increases ranged from $1 to $1.70 per hour and all increases were designed to annually index to
inflation .
</item 5>

<item 6>
Minimum wage in the United States
May 2006 . Retrieved 2013 - 10 - 06 . ^ Jump up to : `` National Occupational Employment and Wage Estimates United States '' . US Bureau of Labor Statistics .
May 2008 . Retrieved 2013 - 10 - 06 . Jump up ^ `` National Occupational Employment and Wage Estimates United States '' . US Bureau of Labor Statistics . May
2009 . Retrieved 2013 - 10 - 06 . External links ( edit ) Minimum wage in the United States at Curlie ( based on DMOZ ) Federal Minimum Wage . United States
Department of Labor Wage and Hour Division . Minimum Wages for Tipped Employees . United States Department of Labor Wage and Hour Division . History of Federal
Minimum Wage United States Department of Labor Wage and Hour Division . U.S. Minimum Wage History . Oregon State University -- Wealth and Poverty ( Anth 484 ) .
Last updated December 26 , 2012 . Maps and charts of minimum wage vs. housing costs - National Low Income Housing Coalition ( advocacy group ) United States
state - related lists List of states and territories of the United States Demographics
</item 6>

<item 7>
Minimum wage in the United States
The minimum wage established by the federal government may be paid to newly hired individuals during their first 90 calendar days of employment , students
employed by colleges and universities , and individuals under 18 years of age . Low - paying occupations , 2006 and 2009 ( edit ) Jobs that a minimum wage is
most likely to directly affect are those that pay close to the minimum . According to the May 2006 National Occupational Employment and Wage Estimates , the
four lowest - paid occupational sectors in May 2006 ( when the federal minimum wage was $5.15 per hour ) were the following : Sector Workers employed Median
wage Mean wage Mean annual Food preparation and serving related occupations 11,029,280 $7.90 $8.86 $18,430 Farming , fishing , and forestry occupations 450,040
$8.63 $10.49 $21,810 Personal care and service occupations 3,249,760 $9.17 $11.02 $22,920 Building and grounds cleaning and maintenance occupations 4,396,250
$9.75 $10.86 $22,580 Two years later , in May 2008 , when the federal minimum wage was $5.
</item 7>

<item 8>
Minimum wage in the United States
However , the Court did not extend this principle to minimum wage laws , considering the latter as interfering with the ability of employers to freely negotiate
wage contracts with employees . In 1933 , the Roosevelt administration made the first attempt at establishing a national minimum wage , when a $0.25 per hour
standard was set as part of the National Industrial Recovery Act . However , in the 1935 court case Schechter Poultry Corp. v. United States ( 295 U.S. 495 ) ,
the US Supreme Court declared the act unconstitutional , and the minimum wage was abolished . In 1938 , the minimum wage was re-established pursuant to the Fair
Labor Standards Act , once again at $0.25 per hour ( $4.78 in 2017 dollars ) . In 1941 , the Supreme Court upheld the Fair Labor Standards Act in United States
v. Darby Lumber Co. , holding that Congress had the power under the Commerce Clause to regulate employment conditions .
</item 8>

<item 9>
Minimum wage in the United States
Federal ( edit ) Type Min . wage ( $ / h ) Notes Tipped $2.13 The Fair Labor Standards Act of 1938 has been requiring a minimum wage of $2.13 for tipped workers
with the expectation that wages plus tips total no less than $7.25 per hour since September 1 , 1991 . The employer must pay the difference if total income does
not add up to $7.25 per hour . Non-tipped $7.25 Per the Fair Minimum Wage Act of 2007 ( FMWA ) since July 24 , 2009 . Youth $4.25 The Fair Labor Standards Act
has since September 1 , 1996 allowed for persons under the age of 20 to be paid $4.25 for the first 90 calendar days of their employment . State ( edit ) The
average US minimum wage per capita ( 2017 ) is $8.49 based on the population size of each state and generally represents the average minimum wage experienced by
a person working in one of the fifty US states . Cities , counties , districts , and territories are not included in the calculation .
</item 9>

<item 10>
Minimum wage in the United States
20th century ( edit ) In 1910 , in conjunction with advocacy work led by Florence Kelley of the National Consumer League , the Women 's Trade Union League (
WTLU ) of Massachusetts under the leadership of Elizabeth Evans took up the cause of minimum wage legislation in Massachusetts . Over the next two years , a
coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state . In 1912 , Massachusetts passed the first
American minimum wage legislation , which established a state commission for recommending non-compulsory minimum wages for women and children . Within eight
years , at least thirteen U.S. states and the District of Columbia would pass minimum wage laws , with pressure being placed on state legislatures by the
National Consumers League in a coalition with other women 's voluntary associations and organized labor . The United States Supreme Court of the Lochner era
consistently invalidated compulsory minimum wage laws . Advocates for these minimum wage laws hoped that they would be upheld under the precedent of Muller v.
Oregon , which upheld maximum working hours laws for women on the grounds that women required special protection that men did not .
</item 10>
=== END ITEM id=5679ee1cc71c461b ===

=== ITEM id=5be4aab0b6bf3fc9 ===
Gold sentences and facts:
Sentence 0: In 1980 , The Shining became the basis for a film adaptation directed by Stanley Kubrick . Kubrick 's vision for the movie differed from King 's significantly in many ways , including the portrayal of the Overlook Hotel . The exteriors of Kubrick 's Overlook were supplied by the Timberline Lodge on the slopes of Mt . Hood in Oregon . Inspiration for the interior sets ( erected at Elstree Studios in England ) came from the 1927 Ahwahnee Hotel in Yosemite National Park . The management of the Timberline Lodge , fearful that guests would refuse to stay in their Room 217 if it were featured in a horror movie , insisted that Kubrick change the haunted room in the film to Room 237 .
  Fact 0.0: The Shining became the basis for a film adaptation in 1980.
  Fact 0.1: The Shining film adaptation was directed by Stanley Kubrick.
  Fact 0.2: Kubrick's vision for the movie differed from Stephen King's significantly in many ways.
  Fact 0.3: Kubrick's vision differed from King's in the portrayal of the Overlook Hotel.
  Fact 0.4: The exteriors of Kubrick's Overlook were supplied by the Timberline Lodge.
  Fact 0.5: The Timberline Lodge is on the slopes of Mt. Hood in Oregon.
  Fact 0.6: Inspiration for the interior sets of the Kubrick film came from the 1927 Ahwahnee Hotel in Yosemite National Park.
  Fact 0.7: The interior sets were erected at Elstree Studios in England.
  Fact 0.8: The management of the Timberline Lodge was fearful that guests would refuse to stay in their Room 217 if it were featured in a horror movie.
  Fact 0.9: The management of the Timberline Lodge insisted that Kubrick change the haunted room in the film to Room 237.

Retrieved items:
<item 1>
The Stanley Hotel
- Roe Emery Maurice L. Albertson , Civil Engineer at CSU Maxwell Abbell , Chicago real estate investor 1995 -- present - Grand Heritage Hotel Group See also (
edit ) Freelan Oscar Stanley The Shining ( novel ) The Shining ( film ) The Shining ( miniseries ) Colonial Revival Architecture Estes Park , CO National
Register of Historic Places listings in Larimer County , Colorado References ( edit ) ^ Jump up to : National Park Service ( 2008 - 04 - 15 ) . `` National
Register Information System '' . National Register of Historic Places . National Park Service . ^ Jump up to : `` Rocky Mountain Legends '' .
LegendsOfAmerica.com . Missing or empty url = ( help ) ; access - date = requires url = ( help ) ^ Jump up to : `` Rocky Mountain National Park - Culture '' .
US-Parks.com . Retrieved 2007 - 05 - 10 . Jump up ^ Turkewitz , Julie ( 3 September 2015 ) . `` Hotel That Inspired ' The Shining ' Builds on Its Eerie Appeal
'' . New York Times . Retrieved 17 September 2015 .
</item 1>

<item 2>
The Stanley Hotel
Stanley holding one of his violins . The sculpture was installed and dedicated on September 29 , 2016 . They used another hotel to film the outside shots and
the Stanley for inside shots . The Shining ( edit ) Cover of The Shining , first edition , 1977 On October 30 , 1974 , horror writer Stephen King and his wife
Tabitha spent one night at the Stanley Hotel during their one year of residency in Boulder , Colorado . They were driving through Estes Park as night approached
when they happened upon the Stanley Hotel . They decided to book a room and , upon checking in , discovered that they were the only overnight guests . `` They
were just getting ready to close for the season , and we found ourselves the only guests in the place -- with all those long , empty corridors . '' He and his
wife were served dinner in an empty dining room accompanied by canned orchestral music . `` Except for our table all the chairs were up on the tables . So the
music is echoing down the hall , and , I mean , it was like God had put me there to hear that and see those things . ''
</item 2>

<item 3>
The Stanley Hotel
Approximately five miles from the entrance to Rocky Mountain National Park , the Stanley offers panoramic views of Lake Estes , the Rockies and especially Long
's Peak . It was built by Freelan Oscar Stanley of Stanley Steamer fame and opened on July 4 , 1909 , catering to the American upper class at the turn of the
century . The hotel and its surrounding structures are listed on the National Register of Historic Places . The Stanley Hotel hosted the horror novelist Stephen
King , serving as inspiration for the Overlook Hotel in his 1977 bestseller The Shining and its 1980 film adaption of the same name , as well as the location
for the 1997 miniseries . Today , it includes a restaurant , spa , and bed - and - breakfast and provides guided tours which feature the history and alleged
paranormal activity of the site .
</item 3>

<item 4>
The Stanley Hotel
Famous guests ( edit ) The Stanley Hotel has hosted the following persons of note : 1934 - Erich Fromm , the German psychoanalyst 1936 - Gov. Alf Landon ( R ,
Kansas ) : opponent to Franklin Delano Roosevelt 's re-election 1974 - Stephen King , author who was inspired to write The Shining from his stay in room 217
1976 - Bob Dylan & Joan Baez , classic rock icons during the Rolling Thunder Revue tour 1994 - HM Emperor Akihito of Japan with consort and crown prince , on a
state visit to the US Ownership History ( edit ) 1908 - 26 - Freelan Oscar Stanley 1926 - 29 - The Stanley Corporation 1929 - 30 - Freelan Oscar Stanley 1930 -
?
</item 4>

<item 5>
The Stanley Hotel
That night , according to King , `` I dreamed of my three - year - old son running through the corridors , looking back over his shoulder , eyes wide ,
screaming . He was being chased by a fire - hose . I woke up with a tremendous jerk , sweating all over , within an inch of falling out of bed . I got up , lit
a cigarette , sat in a chair looking out the window at the Rockies , and by the time the cigarette was done , I had the bones of The Shining firmly set in my
mind . '' The Shining was published in 1977 and became the third great success of King 's career after Carrie and ' Salem 's Lot . The Overlook Hotel - the
fictional hotel in King 's book inspired by the Stanley - is an evil entity haunted by its many victims . Room 217 of the Overlook features prominently in the
novel , having been the room where King spent the night at the Stanley . This is the room on the second floor in the center of the west wing with a balcony
overlooking the south terrace . Room 217 remains the hotel 's most requested accommodation .
</item 5>

<item 6>
The Stanley Hotel
Contents ( hide ) 1 History 2 Gallery 3 Architecture 3.1 Main Building 3.2 Concert Hall 3.3 The Lodge 3.4 The Hedge Maze 4 The Shining 5 Film location and arts
venue 6 Haunted reputation 7 Famous Guests 8 Ownership History 9 See also 10 References 11 External links History ( edit ) In 1903 , the Yankee steam - powered
car inventor Freelan Oscar Stanley ( 1849 - 1940 ) was stricken with a life - threatening resurgence of tuberculosis . The most highly recommended treatment of
the day was fresh , dry air with lots of sunlight and a hearty diet . Therefore , like many `` lungers '' of his day , Stanley resolved to take the curative air
of Rocky Mountain Colorado . He and Flora arrived in Denver in March and , in June , decided to spend the rest of the summer in the mountains , in Estes Park .
Over the course of the season , Stanley 's health improved dramatically . Impressed by the beauty of the valley and grateful for his recovery , he decided to
return every year .
</item 6>

<item 7>
The Stanley Hotel
The Hotel has also been used as a filming location for other movies and TV shows ; most notably , as the fictional `` Hotel Danbury '' of Aspen , CO , in the
1994 film Dumb and Dumber . From 2013 to 2015 , the hotel property hosted the Stanley Film Festival , an independent horror film festival operated by the Denver
Film Society , held in early May . The festival featured screenings , panels , student competitions , audience awards and receptions . The Stanley Film Festival
was put on hiatus in 2016 , and canceled for 2017 . The historic Stanley Concert Hall serves as venue for various musical groups such as country - punk band
Murder By Death which has performed a Shining - themed series of concerts in the space . Haunted reputation ( edit ) Despite a peaceful early history , in the
years following the publication of The Shining , The Stanley Hotel has gained a reputation among paranormal investigators for frequent activity . The hotel
offers guided `` Ghost Tours '' to guests and visitors which feature spaces reputed to be exceptionally active . The hotel has also served as a location for
numerous paranormal investigation shows such as Ghost Hunters and Ghost Adventures .
</item 7>

<item 8>
The Stanley Hotel
Water was supplied by the Black Canyon Creek which was dammed in 1906 . The style of the campus is so - called Colonial Revival . Although rare in the western
United States , F.O. Stanley chose the Colonial Revival for its fashionable popularity in New England where he had already designed his own home and a social
club in the style . The hotel 's clientele would presumably , like the Stanleys , have identified the style with New England respectability and sophistication
in contrast to the rusticity of the surrounding town . At one time , Stanley planned to build another , more economical hotel in Estes Park as well as a
headquarters and residence for the superintendent of the Rocky Mountain National Park in the same style , to harmonize with his grand hotel . While the forms
and layouts of the buildings are suited to their modern uses , their ornamentation exhibits the stylistic hallmarks of Georgian or Federal architecture from the
staunch symmetry of the south elevation to the cupola , dormers , Palladian windows , side - and fan - lights , scroll brackets and `` swan 's neck '' pediments
that articulate the exterior .
</item 8>

<item 9>
The Stanley Hotel
Room 217 remains the hotel 's most requested accommodation . Film location and arts venue ( edit ) In 1980 , The Shining became the basis for a film adaptation
directed by Stanley Kubrick . Kubrick 's vision for the movie differed from King 's significantly in many ways , including the portrayal of the Overlook Hotel .
The exteriors of Kubrick 's Overlook were supplied by the Timberline Lodge on the slopes of Mt . Hood in Oregon . Inspiration for the interior sets ( erected at
Elstree Studios in England ) came from the 1927 Ahwahnee Hotel in Yosemite National Park . The management of the Timberline Lodge , fearful that guests would
refuse to stay in their Room 217 if it were featured in a horror movie , insisted that Kubrick change the haunted room in the film to Room 237 . In 1997 The
Shining TV series was produced , with the Stanley Hotel as the primary shooting location . A playhouse version of the main building which adorned the lawn of
the Overlook Hotel in the series can be viewed in the basement of the Stanley .
</item 9>

<item 10>
The Stanley Hotel
The hedge maze ( edit ) In 2015 , the open area in front of the hotel , originally a driveway for Stanley Steamers and a promenade for guests to enjoy mountain
views , was replaced with a hedge maze . This non-historic feature was added to evoke the hotel 's connection to The Shining . Although a hedge maze features
prominently in Stanley Kubrick 's version of The Shining , no such feature can be found in Stephen King 's novel . The lawn of the Overlook Hotel , as King
imagined it , was adorned with topiary animals and it was set that way for the shooting of the miniseries in 1996 . Historically , neither feature existed at
the Stanley . The design was the result of a competition , chosen from amongst 300 entrants from across the globe . The winning submission was designed by New
York architect Mairim Dallaryan Standing . In 2016 , John Cullen , owner of the Stanley Hotel , announced a competition for a sculpture to be the centerpiece of
the terrace in front of the hotel . Sculptors Sutton Betti and Daniel Glanz won the competition with a sculpture of F.O.
</item 10>
=== END ITEM id=5be4aab0b6bf3fc9 ===
