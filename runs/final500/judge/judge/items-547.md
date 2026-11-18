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

This file has 11 items.

=== ITEM id=e2c525079ae8ec0f ===
Gold sentences and facts:
Sentence 0: Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
  Fact 0.0: Rolf Martin Theodor Dahlgren was born on 7 July 1932.
  Fact 0.1: Rolf Martin Theodor Dahlgren died on 14 February 1987.
  Fact 0.2: Rolf Martin Theodor Dahlgren was a Swedish-Danish botanist.
  Fact 0.3: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen.
  Fact 0.4: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen from 1973 until his death.
Sentence 1:  The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.
  Fact 1.0: The university has 23,473 undergraduate students.
  Fact 1.1: The university has 17,398 postgraduate students.
  Fact 1.2: The university has 2,968 doctoral students.
  Fact 1.3: The university has over 9,000 employees.

Retrieved items:
<item 1>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 1>

<item 2>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 2>

<item 3>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 3>

<item 4>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 4>

<item 5>
Richard W. Aldrich
Richard Warren Aldrich is an American neuroscientist who is currently the Karl Folkers Chair of Interdisciplinary Medical Research and Professor of Neurobiology
in the School of Biological Sciences at the University of Texas.

 Graduating from the University of Arizona with a BS in Biological Sciences (1975), Aldrich went on to earn a Ph. D. in Neuroscience from Stanford University
(1980).

 After completing a post-doctoral fellowship in physiology at Yale University under the direction of W. Knox Chandler and Charles F. Stevens, Aldrich was hired
to teach at Yale, instructing in the Department of Molecular Neurobiology.

 In 1985, Aldrich returned to Stanford to teach neurobiology and physiology, eventually serving as Chair of the Department of Molecular and Cellular Physiology
from 2001 to 2004.

 He was an Investigator in the Howard Hughes Medical Institute from 1990 until 2006.

 Aldrich's research has focused on the "molecular mechanisms of ion channel function and their role in electrical signaling."

 Aldrich has served as the President of the Society of General Physiologists and the Biophysical Society.
</item 5>

<item 6>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 6>

<item 7>
Rolf Dahlgren
Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
</item 7>

<item 8>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 8>

<item 9>
Dimitrije Đorđević (historian)
Đorđević was awarded his doctorate in 1962.

 In 1970, Đorđević took up a position as a Full Professor of History at the University of California, Santa Barbara, joining a strong faculty in European
History including Joachim Remak, Leonard Marsak, and C. Warren Hollister.

 He was elected a member of the Serbian Academy of Science and Arts in 1985.

 A popular undergraduate lecturer and graduate mentor at the University of California, Santa Barbara, in 1992 many of his former students contributed to his
"Festschrift" entitled "Scholar, Patriot, Mentor: Historical Essays in Honor of Dimitrije Djordjevic".

 In retirement, Đorđević published his autobiography, "Scars and Memory: Four Lives in One Lifetime", describing his World War II and post World War II
experiences.

 Professor Đorđević died in Santa Barbara on March 5, 2009.
</item 9>

<item 10>
Gertrud Dahlgren
Gertrud Dahlgren (1931–2009) was a Swedish botanist.

 She was born at Klippan in Scania, and after graduating from university at Helsingborg went to the University of Lund for graduate studies.

 There she obtained her M.Sc.

 in chemistry and biology.

 She was married to fellow botanist Rolf Dahlgren (1932–1987), who was killed in a car accident.

 The couple had three children, Susanna, Helena and Fredrik.

 She died in December 2009 at the age of 78.
</item 10>
=== END ITEM id=e2c525079ae8ec0f ===

=== ITEM id=d29dd872d058822e ===
Gold sentences and facts:
Sentence 0: Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in England to Australian parents.
  Fact 0.0: Ralph Foster Smart was born on 27 August 1908.
  Fact 0.1: Ralph Foster Smart died on 12 February 2001.
  Fact 0.2: Ralph Foster Smart was a film producer.
  Fact 0.3: Ralph Foster Smart was a television producer.
  Fact 0.4: Ralph Foster Smart was a director.
  Fact 0.5: Ralph Foster Smart was a writer.
  Fact 0.6: Ralph Foster Smart was born in England.
  Fact 0.7: Ralph Foster Smart's parents were Australian.
Sentence 1: Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.
  Fact 1.0: Randolph Severn 'Trey' Parker III was born on October 19, 1969.
  Fact 1.1: Randolph Severn 'Trey' Parker III is an American actor.
  Fact 1.2: Randolph Severn 'Trey' Parker III is an animator.
  Fact 1.3: Randolph Severn 'Trey' Parker III is a writer.
  Fact 1.4: Randolph Severn 'Trey' Parker III is a director.
  Fact 1.5: Randolph Severn 'Trey' Parker III is a producer.
  Fact 1.6: Randolph Severn 'Trey' Parker III is a singer.
  Fact 1.7: Randolph Severn 'Trey' Parker III is a songwriter.

Retrieved items:
<item 1>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 1>

<item 2>
Ralph Smart
Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in
England to Australian parents.
</item 2>

<item 3>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 3>

<item 4>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 4>

<item 5>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 5>

<item 6>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 6>

<item 7>
Trey Parker
Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.

 He is known for co-creating "South Park" (1997–present) along with his creative partner Matt Stone, as well as co-writing and co-directing the Tony
Award-winning musical "The Book of Mormon" (2011).

 Parker was interested in film and music as a child, and attended the University of Colorado, Boulder following high school, where he met Stone.

 The two collaborated on various short films, and starred in a feature-length musical, titled "Cannibal!

 The Musical" (1993).
</item 7>

<item 8>
Curtain Up
Curtain Up is a 1952 British film directed by Ralph Smart, written by Jack Davies and Michael Pertwee.

 It is based on the play "On Monday Next" by Philip King.
</item 8>

<item 9>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 9>

<item 10>
Rodger Bumpass
Rodger Bumpass (born November 20, 1951) is an American actor and voice actor.

 He is best known for his long-running role as Squidward Tentacles on the American animated television series "SpongeBob SquarePants".

 He voices many other characters on the show, including Dr. Forrest and various anchovies.

 He also voiced The Chief in the animated series "Where on Earth Is Carmen Sandiego?

", and Mr. Besser, the school principal in the animated series "The Kids from Room 402".

 Bumpass has many other credits in animated films, animated television series, and video games.
</item 10>
=== END ITEM id=d29dd872d058822e ===

=== ITEM id=0394c37077ea99e4 ===
Gold sentences and facts:
Sentence 0: Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic series.
  Fact 0.0: Levni Yilmaz was born in 1973.
  Fact 0.1: Levni Yilmaz is a San Francisco based independent film maker.
  Fact 0.2: Levni Yilmaz is an artist.
  Fact 0.3: Levni Yilmaz is a publisher.
  Fact 0.4: Levni Yilmaz is best known for his 'Tales of Mere Existence' animated comic series.
Sentence 1: Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.
  Fact 1.0: Pamela B. Green is American.
  Fact 1.1: Pamela B. Green is a film producer.
  Fact 1.2: Pamela B. Green is a film director.
  Fact 1.3: Pamela B. Green is known for her work in feature film titles.
  Fact 1.4: Pamela B. Green is known for her work in motion graphics.

Retrieved items:
<item 1>
İsmet Ergün
İsmet Ergün (born 1950) is a Berlin artist and stage designer of Turkish provenance.

 Alongside her theatre work she has worked in the world of cinema as an art director and as a production designer.

 The short film which she directed, and for which she herself wrote the screenplay, was singled out for commendation at the 2007 Locarno Film Festival: it also
won the "Best German Film 2007" at the later the same year.
</item 1>

<item 2>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 2>

<item 3>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 3>

<item 4>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 4>

<item 5>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 5>

<item 6>
Pamela B. Green
Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.

 She is the director and producer of the documentary "Be Natural: The Untold Story of Alice Guy-Blache".
</item 6>

<item 7>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 7>

<item 8>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 8>

<item 9>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 9>

<item 10>
Lev Yilmaz
Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic
series.
</item 10>
=== END ITEM id=0394c37077ea99e4 ===

=== ITEM id=1dec5a37a8e475ed ===
Gold sentences and facts:
Sentence 0: Dieffenbachia is a genus of tropical flowering plants in the family Araceae.
  Fact 0.0: Dieffenbachia is a genus of flowering plants.
  Fact 0.1: Dieffenbachia is a genus of tropical flowering plants.
  Fact 0.2: Dieffenbachia is in the family Araceae.
Sentence 1: Carlina is a genus of flowering plants in the aster family, Asteraceae.
  Fact 1.0: Carlina is a genus of flowering plants.
  Fact 1.1: Carlina is in the aster family.
  Fact 1.2: Carlina is in the family Asteraceae.

Retrieved items:
<item 1>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 1>

<item 2>
Carlina
Carlina is a genus of flowering plants in the aster family, Asteraceae.

 It is distributed from Madeira and the Canary Islands across Europe and northern Africa to Siberia and northwestern China.
</item 2>

<item 3>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 3>

<item 4>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 4>

<item 5>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 5>

<item 6>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 6>

<item 7>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 7>

<item 8>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 8>

<item 9>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 9>

<item 10>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 10>
=== END ITEM id=1dec5a37a8e475ed ===

=== ITEM id=7c6c2f942d099350 ===
Gold sentences and facts:
Sentence 0: Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the electronica band Republica.
  Fact 0.0: Samantha Marie Sprackling was born on 3 June 1968.
  Fact 0.1: Samantha Marie Sprackling was born in Lagos, Nigeria.
  Fact 0.2: Samantha Marie Sprackling is more commonly known as Saffron.
  Fact 0.3: Samantha Marie Sprackling is a singer.
  Fact 0.4: Samantha Marie Sprackling is a songwriter.
  Fact 0.5: Samantha Marie Sprackling is an actress.
  Fact 0.6: Samantha Marie Sprackling is the lead singer of the electronica band Republica.
Sentence 1: Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.
  Fact 1.0: Myles Richard Kennedy was born Myles Richard Bass.
  Fact 1.1: Myles Richard Kennedy was born on November 27, 1969.
  Fact 1.2: Myles Richard Kennedy is an American musician.
  Fact 1.3: Myles Richard Kennedy is an American singer.
  Fact 1.4: Myles Richard Kennedy is an American songwriter.
  Fact 1.5: Myles Richard Kennedy is best known as the lead vocalist and rhythm guitarist of the rock band Alter Bridge.
  Fact 1.6: Myles Richard Kennedy is the lead vocalist in guitarist Slash's backing band.
  Fact 1.7: Slash's backing band is known as Myles Kennedy and the Conspirators.
Sentence 2:  A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several artists, and has been involved with several projects throughout his career.
  Fact 2.0: Myles Richard Kennedy is a former guitar instructor.
  Fact 2.1: Myles Richard Kennedy is from Spokane, Washington.
  Fact 2.2: Myles Richard Kennedy has worked as a session musician.
  Fact 2.3: Myles Richard Kennedy has worked as a songwriter.
  Fact 2.4: Myles Richard Kennedy has made studio appearances with several artists.
  Fact 2.5: Myles Richard Kennedy has made live appearances with several artists.
  Fact 2.6: Myles Richard Kennedy has been involved with several projects throughout his career.

Retrieved items:
<item 1>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 1>

<item 2>
Myles Kennedy
Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm
guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.

 A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several
artists, and has been involved with several projects throughout his career.
</item 2>

<item 3>
Bent to Fly
"Bent to Fly" is a song by American hard rock guitarist Slash, featuring vocalist Myles Kennedy and backing band The Conspirators.

 Written by Slash and Kennedy, it was released as the second single from the guitarist's third solo album (the second with Kennedy and The Conspirators), "World
on Fire".

 The song was used as the theme song for the 2014 National Rugby League Finals series, and Slash performed the song live at ANZ Stadium as pre-show
entertainment for the league's grand final that year.
</item 3>

<item 4>
List of songs recorded by Myles Kennedy
Myles Kennedy is an American rock musician.

 Born in Boston, Massachusetts, he originally began his musical career in Spokane, Washington as the guitarist in jazz band Cosmic Dust.

 After recording 1991's "Journey", he left to form alternative rock band Citizen Swing in 1992.

 The group released two albums, 1992's "Cure Me with the Groove" and 1996's "Deep Down", on both of which Kennedy was co-credited for songwriting.

 Kennedy's next band was The Mayfield Four, which he co-founded with guitarist Craig Johnson, bassist Marty Meisner and drummer Zia Uddin in 1996.

 The frontman wrote most of the material on the band's 1998 debut "Fallout" and all of the songs on 2001's "Second Skin".
</item 4>

<item 5>
Apocalyptic Love World Tour
The Apocalyptic Love World Tour is the second concert tour by ex-Guns N' Roses lead guitarist Slash as a solo artist.

 The tour features the same backing band that toured with Slash during his first solo tour, though this time the act is billed as "Slash featuring Myles Kennedy
and The Conspirators".

 The tour is in support of Slash's second solo album "Apocalyptic Love", which features Myles Kennedy handling lead vocal duties on all songs, as well as his
touring band members: bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 5>

<item 6>
World on Fire (Slash song)
"World on Fire" is a song by American hard rock guitarist Slash, featuring vocalist Myles Kennedy and backing band The Conspirators.

 Written by Slash and Kennedy, it is the title track of the guitarist's third solo album (the second with Kennedy and The Conspirators) "World on Fire".

 Released as the album's lead single, "World on Fire" topped the US "Billboard" Mainstream Rock Songs chart.
</item 6>

<item 7>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 7>

<item 8>
Myles Kennedy discography
The full discography of rock musician Myles Kennedy consists of eleven studio albums, two concert films, four live albums, two extended plays, and thirteen
singles in total, in addition to eleven studio tracks that he has appeared on as a featured artist, one of which was a single.

 Born in Boston on November 27, 1969, Kennedy is currently a member of the rock band Alter Bridge, with whom he has released four studio albums, two concert
films, and several singles.

 He is also the frontman of Slash's touring group, and with Slash he has released a live album, "Live in Manchester", the first of a series of live albums
released throughout the summer of 2010, and "", another live album released in 2011.

 In 2012, he released a collaboration studio album with Slash titled "Apocalyptic Love", which is billed to Slash featuring Myles Kennedy and the Conspirators,
as well as the 2014 followup titled "World on Fire".

 With The Mayfield Four, he released two studio albums, two extended plays, and four singles; with Citizen Swing, two studio albums; and with Cosmic Dust, one
studio album.
</item 8>

<item 9>
The material surveys rock albums, bands, tours, and Myles Kennedy’s career. Highlights include Juke Kartel’s 2010 album *Levolution*; OneRepublic’s
five-continent Native Tour; Van Halen, Journey, Kiss, U2, Bon Jovi, and other major tours; and 2010’s Blackdiamondskye. It traces Kennedy from jazz band Cosmic
Dust through Citizen Swing and The Mayfield Four to Alter Bridge and Slash’s Conspirators, including his
</item 9>

<item 10>
World on Fire World Tour
The World on Fire World Tour is the third concert tour by lead guitarist Slash as a solo artist, which started in July 2014 and was scheduled to resume in late
2015, in support of Slash's third solo album "World on Fire".

 The tour features the same backing band that performed with Slash during his two last tours, billed as "Slash featuring Myles Kennedy and The Conspirators",
featuring Myles Kennedy handling lead vocal, bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 10>
=== END ITEM id=7c6c2f942d099350 ===

=== ITEM id=d0d37aad42ee793d ===
Gold sentences and facts:
Sentence 0:  Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
  Fact 0.0: Cacareco's candidacy inspired the Rhinoceros Party of Canada.
  Fact 0.1: The Rhinoceros Party of Canada was nominally led by the rhinoceros Cornelius the First.
Sentence 1: The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.
  Fact 1.0: The Rhinoceros Party's French name is 'Parti Rhinoceros'.
  Fact 1.1: The Rhinoceros Party was a registered political party in Canada.
  Fact 1.2: The Rhinoceros Party was a registered political party in Canada from the 1960s to the 1990s.

Retrieved items:
<item 1>
History of the United States Republican Party
The Republican Party, also commonly called the GOP (for "Grand Old Party"), is one of the world's oldest extant political parties.

 It is the second oldest existing political party in the United States after its primary rival, the Democratic Party.

 It emerged in 1854 to combat the Kansas–Nebraska Act, an act that dissolved the terms of the Missouri Compromise and allowed slave or free status to be decided
in the territories by popular sovereignty.

 The Party had almost no presence in the Southern United States, but by 1858 in the North it had enlisted former Whigs and former Free Soil Democrats to form
majorities in nearly every Northern state.
</item 1>

<item 2>
Aboriginal Peoples Party of Canada
The Aboriginal Peoples Party of Canada (APP) is a Canadian political party that was founded in 2005.

 The party was conceived by University of Lethbridge student Myron Wolf Child.

 It held its founding meeting on August 21, 2005, in St. Albert, Alberta.

 The APP was headed by interim leader Bill Montour, a former Chief of the Six Nations Reserve in Ontario.

 In October 2005, the party had 122 confirmed members, just less than half the number needed to register as an official party in Canada.

 In late October 2005, the APP sought to unite with the First Peoples National Party of Canada which also had fewer than the number of confirmed members needed
to become a registered political party.

 The First Peoples National Party of Canada became an eligible political party on December 6, 2005.

 Whether or not this was accomplished through a merger with the APP is unclear.

 If this is the case, the APP no longer exists, but, as one of the stipulations of a possible merger was that a name for the new party would be determined at
the party's first convention, it is possible that the name Aboriginal Peoples Party of Canada will come into usage again.
</item 2>

<item 3>
Revolutionary Communist Party of Canada
The Revolutionary Communist Party of Canada, or RCP Canada, is a revolutionary communist party oriented around Marxism-Leninism-Maoism.

 The creation of the organizational stage of the Party was adopted at what was called the "Revolutionary Communist Conference", which was held in Montreal,
Quebec in November 2000 by activists and former members of the labour union movements and youth organizations of the 1960s, 1970s and 1980s, who felt that the
revolutionary situation in Canada warranted the creation of a Party dedicated to a communist revolution.

 The party was initially called the Revolutionary Communist Party (Organizing Committees) (RCP(OC)).

 At this conference, participants adopted the Party's first Draft Programme.

 Because of the location of the conference, the majority of the founding members were French-speaking Quebecers, and the Party began an extensive effort to
reach out to the rest of Canada, starting with the Canadian Revolutionary Congress held in November 2006 in Toronto.

 It is not recognized as a political party by Elections Canada, because the party rejects what it calls the "bourgeois electoral system" and doesn't seek
recognition by the State.
</item 3>

<item 4>
The New Canada
The New Canada is a Canadian political literature book written by Reform Party of Canada founder and leader Preston Manning and published by Macmillan Canada.

 The book explains the personal, religious, and political life of Preston Manning and explains the roots and beliefs of the Reform Party.

 At the time of its publishing in 1992, Reform had become a popular populist conservative party in Western Canada after the mainstream Progressive Conservative
Party of Canada was collapsing in support and in 1991 decided to expand eastward into Ontario and the Maritimes.

 One year later the PC party collapsed in the 1993 federal election, allowing the Reform Party to make political history in Canada, deplacing the PCs as the
dominant conservative party in Canada.

 Reform, later renamed the Canadian Alliance, merged with the PC Party in 2003, to form a united right-wing alternative to the governing Liberal Party of
Canada, named the Conservative Party of Canada which has dropped many of the populist themes that the Reform Party had.
</item 4>

<item 5>
Civil Human Rights Front
Civil Human Rights Front or CHRF () is an organisation that focuses on the issues of Hong Kong politics and livelihood, affiliates almost all the pan-democratic
camps in Hong Kong.

 Forty-eight NGOs and political groups have been involved in the organisation as of January 2006.

 The most well-known event held by the CHRF is the Hong Kong 1 July marches.
</item 5>

<item 6>
Cacareco
Cacareco (1954-1962) was a female black rhinoceros exhibited in Brazilian zoos.

 She became famous as a candidate for the 1958 São Paulo city council elections with the intention of protesting against political corruption.

 Electoral officials did not accept Cacareco's candidacy, but she eventually won 100,000 votes, more than any other party in that same election (which was also
marked by rampant absenteeism).

 Today, the term "voto Cacareco" (Cacareco vote) is commonly used to describe protest votes in Brazil.

 Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
</item 6>

<item 7>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 7>

<item 8>
Rhinoceros Party of Canada (1963–93)
The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.

 Operating within the tradition of political satire, the Rhinoceros Party's basic credo, their so-called primal promise, was "a promise to keep none of our
promises".

 They then promised outlandishly impossible schemes designed to amuse and entertain the voting public.
</item 8>

<item 9>
LPRadicals
LPRadicals is a caucus formed in 2006 within the United States Libertarian Party by Susan Hogarth and other party members who opposed removal of much of the
material in the party platform during the 2006 national party convention.

 The caucus lists four points as "key strategic principles in furthering the work of the Libertarian Party toward, as our platform describes it, 'a world set
free in our lifetime'."

 These are: Rights Are Utilitarian, Radical Abolitionism, Principled Populism, and No Particular Order (for removal of government policies).

 The caucus was active at the 2008 and 2010 Libertarian National Conventions.

 The radical caucus was revived and was extraordinarily active during the 2016 Libertarian National Convention.
</item 9>

<item 10>
National Advancement Party of Canada
The National Advancement Party of Canada (French: "Parti de l'Essor National du Canada" ) is a minor federal political party in Canada.

 The party describes itself as for "advancement of the Canadian people and Canada".

 The party was formed in 2014 and registered with Elections Canada in 2015, initially under the name Democratic Advancement Party of Canada.

 The party is headquartered in Calgary, Alberta and led by writer Stephen Garvey.
</item 10>
=== END ITEM id=d0d37aad42ee793d ===

=== ITEM id=03fcd26440a5320b ===
Gold sentences and facts:
Sentence 0: Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.
  Fact 0.0: Tender Fictions is a 1996 autobiographical documentary film.
  Fact 0.1: Tender Fictions was directed by American experimental filmmaker Barbara Hammer.
Sentence 1: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
  Fact 1.0: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
Sentence 2:  It was produced and directed by Wayne Coles-Janess.
  Fact 2.0: Bougainville – Our Island Our Fight was produced by Wayne Coles-Janess.
  Fact 2.1: Bougainville – Our Island Our Fight was directed by Wayne Coles-Janess.

Retrieved items:
<item 1>
Wayne Coles-Janess
Wayne Coles-Janess is an Australian producer, writer and director of drama and documentary film and TV programs.

 Based in Melbourne, Australia, he has produced documentaries about frontier places in the country.

 He has also made some documentaries in several international locations, including during times of war.
</item 1>

<item 2>
The entries survey documentaries, films, TV, and film-related concepts: Los Angeles hip-hop, Aboriginal country music, Latin American politics, oceans, nerd
culture, Iraqi life and war, Bougainville, education, disability, poetry, surfing, and family relationships. Featured works include *Gasland*, *Baghdad ER*,
*War/Dance*, *First Life*, and *The Coconut Revolution*. Several received major awards or nominations. The material also profiles filmmaker Wayne Coles-Janess
and defines
</item 2>

<item 3>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 3>

<item 4>
The entries survey theatre, film and literary figures/works: Boucicault’s 1860 Irish melodrama *The Colleen Bawn*; director Jules Wright and actress-filmmaker
Laura Bayley; films including *Summer Holiday*, *Trio*, *Julia*, *The King and I*, and *Anna and the King of Siam*; and plays by Lillian Hellman, Steven Dietz,
Aphra Behn, John Burgoyne, and Thornton
</item 4>

<item 5>
The Coconut Revolution
The Coconut Revolution is a 2001 multi-award winning documentary film about the struggle of the indigenous peoples of Bougainville Island during the
Bougainville Civil War.

 The movement is described as the "world's first successful eco-revolution" and has drawn parallels with the conflict depicted in the 2009 film, "Avatar".
</item 5>

<item 6>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 6>

<item 7>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 7>

<item 8>
The entries cover filmmakers, artists and films. Takeshi Fukunaga’s debut, *Out of My Hand* (2015), won the LA Film Festival’s top prize and earned an
Independent Spirit nomination. Hou Hsiao-hsien, a Taiwan New Wave auteur, won Venice’s Golden Lion and Cannes’ Best Director award; his films include *The
Assassin*, *Dust in the Wind* and *The Time to Live and the Time to Die*. Cinematographer Mark Lee
</item 8>

<item 9>
Bougainville – Our Island Our Fight
Bougainville – Our Island Our Fight is a 1998 Australian documentary film.

 It was produced and directed by Wayne Coles-Janess.
</item 9>

<item 10>
The Boys in Company C
The Boys in Company C, directed by Sidney J. Furie, starring Stan Shaw, Andrew Stevens, Craig Wasson, James Canning, and Michael Lembeck, is a 1978 film about
United States Marine Corps recruits preparing for duty, and their subsequent combat in the Vietnam War.

 It was among the first Vietnam War films to appear after the Vietnam Era, and was also the first role for R. Lee Ermey of "Full Metal Jacket" fame.

 "The Boys in Company C" is the first in Furie's Vietnam War motion picture trilogy, followed by 2001's "Under Heavy Fire" and 2006's "The Veteran".
</item 10>
=== END ITEM id=03fcd26440a5320b ===

=== ITEM id=cbe84b706d07829d ===
Gold sentences and facts:
Sentence 0: Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.
  Fact 0.0: Bumble and bumble Products, LLC is one of 27 brands.
  Fact 0.1: The 27 brands are owned by Estée Lauder Companies, Inc.
  Fact 0.2: Bumble and bumble Products, LLC is owned by Estée Lauder Companies, Inc.
Sentence 1:  The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown Manhattan, New York City.
  Fact 1.0: Estée Lauder Companies, Inc. owns a diverse portfolio of brands.
  Fact 1.1: The brand portfolio is distributed internationally through retail channels.
  Fact 1.2: The brand portfolio is distributed internationally through digital commerce.
  Fact 1.3: Estée Lauder Companies, Inc. has its headquarters in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
New York Transportation Company
The New York Transportation Company (originally New York Electrical Vehicle Transportation Company) was a company incorporated in New Jersey in 1899 as the 'New
York Electrical Vehicle Transportation Company' which changed its name to the "New York Transportation Company" in 1902 and was wound up in 1936.

 From 1922 it was controlled by the 'Fifth Avenue Bus Securities Corporation' which was incorporated in Delaware in 1922 which was controlled by The Omnibus
Corporation also incorporated in Delaware.
</item 1>

<item 2>
Buena Vista (brand)
Buena Vista is a brand name which was historically often used for divisions and subsidiaries of The Walt Disney Company, whose primary studios, the Walt Disney
Studios, are located on Buena Vista Street in Burbank, California.

 (The studio lot is also home to the Company's corporate headquarters, the Team Disney Burbank building.)

 The logos for the various Buena Vista brands featured the "Buena Vista" wordmark superimposed over the Disney Castle image to signal the affiliation between
Buena Vista and Disney.

 The name literally means "good view" in Spanish.
</item 2>

<item 3>
Papa Murphy's
Papa Murphy's, a business based in Vancouver, Washington, United States, is a take-and-bake pizza company.

 It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).

 The company and its franchisees operate more than 1,300 outlets in the United States and Canada.

 Papa Murphy's is the fifth-largest pizza chain in the United States.
</item 3>

<item 4>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 4>

<item 5>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 5>

<item 6>
Bell Labs
Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company,
owned by Finnish company Nokia.

 Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
</item 6>

<item 7>
Graph query row:
entity: Bumble and bumble
connection: ['Bumble and bumble', 'Estée Lauder Companies, Inc.']
relationships: ['PART_OF']
organization: Estée Lauder Companies, Inc.
headquarters_city: None
</item 7>

<item 8>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 8>

<item 9>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 9>

<item 10>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
</item 10>
=== END ITEM id=cbe84b706d07829d ===

=== ITEM id=a82357c9ff61efae ===
Gold sentences and facts:
Sentence 0:  The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
  Fact 0.0: The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck.
  Fact 0.1: Dave Brubeck was also one of the company's investors.
  Fact 0.2: The label is known more for its recordings of comedian Lenny Bruce.
  Fact 0.3: The label is known more for its recordings of jazz musician Vince Guaraldi.
  Fact 0.4: The label is known more for its recordings of the rock band Creedence Clearwater Revival.
Sentence 1: Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
  Fact 1.0: Vincent Anthony Guaraldi was born on July 17, 1928.
  Fact 1.1: Vincent Anthony Guaraldi died on February 6, 1976.
  Fact 1.2: Vincent Anthony Guaraldi was born Vincent Anthony Dellaglio.
  Fact 1.3: Vincent Anthony Guaraldi was an American jazz pianist.
  Fact 1.4: Vincent Anthony Guaraldi was noted for his innovative compositions and arrangements.
  Fact 1.5: Vincent Anthony Guaraldi was noted for composing music for animated television adaptations of the 'Peanuts' comic strip.
  Fact 1.6: Vincent Anthony Guaraldi performed on piano as a member of Cal Tjader's late 50s ensemble.
  Fact 1.7: Vincent Anthony Guaraldi had a solo career.
  Fact 1.8: Vincent Anthony Guaraldi's solo career included the radio hit 'Cast Your Fate to the Wind'.

Retrieved items:
<item 1>
Vince Guaraldi
Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions
and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of
Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
</item 1>

<item 2>
The Best of Creedence Clearwater Revival
The Best of Creedence Clearwater Revival is a compilation album by American rock band Creedence Clearwater Revival, released in 1977.

 The album features all the tracks of Chronicle which was released a year earlier in 1976, as well as "Good Golly Miss Molly", "Born on the Bayou", "Cotton
Fields", "Hello Mary Lou", "The Midnight Special", "Walk on the Water", and "Night Time Is the Right Time" which would appear on the 1986 release "Chronicle,
Vol.

 2", and "Bootleg" which does not appear on either volume.
</item 2>

<item 3>
The Long Road Home
The Long Road Home: The Ultimate John Fogerty/Creedence Collection is a compilation album by American roots rock singer-songwriter John Fogerty, released on
November 1, 2005, by Fantasy Records.

 It compiles songs from Fogerty's solo career and his band Creedence Clearwater Revival.

 The title refers to Fogerty's return to Fantasy Records, after a lengthy stint with Warner Bros.

 Records and a brief stint with DreamWorks Records.
</item 3>

<item 4>
Creedence Clearwater Revival (album)
Creedence Clearwater Revival is the debut studio album by the American rock band Creedence Clearwater Revival, released in 1968.
</item 4>

<item 5>
John Fogerty (album)
John Fogerty is the second solo studio album by former Creedence Clearwater Revival vocalist/guitarist John Fogerty, released in 1975.

 It was released by Asylum Records in the United States and Fantasy Records internationally.

 As with the Creedence Clearwater Revival records, the album consists of a mix of originals and cover songs.

 Although the album is eponymously titled, Fogerty himself refers to it as "Old Shep"; Shep was the name of his dog, who appears on the cover with him.
</item 5>

<item 6>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 6>

<item 7>
Fantasy Records
Fantasy Records is an American record company and label founded by brothers Max and Sol Weiss in 1949.

 The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known
more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
</item 7>

<item 8>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 8>

<item 9>
Jerry Beck
Peabody and Sherman", DreamWorks' "Madagascar", and "Bee Movie".

 Beck is also an entertainment industry consultant for TV and home entertainment productions and releases related to classic cartoons and operates the blog
"Cartoon Research."

 He appears frequently as a documentary subject and audio commentator on releases of A&E's "Cartoons Go to War" as well as DVD collections of "Looney Tunes",
"Popeye the Sailor", and "Woody Woodpecker" cartoons, on which he serves as consultant and curator.
</item 9>

<item 10>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 10>
=== END ITEM id=a82357c9ff61efae ===

=== ITEM id=594ea96b1b810d72 ===
Gold sentences and facts:
Sentence 0: Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.
  Fact 0.0: Claudio Giovanni Antonio Monteverdi was baptized on 15 May 1567.
  Fact 0.1: Claudio Giovanni Antonio Monteverdi died on 29 November 1643.
  Fact 0.2: Claudio Giovanni Antonio Monteverdi was an Italian composer.
  Fact 0.3: Claudio Giovanni Antonio Monteverdi was an Italian string player.
  Fact 0.4: Claudio Giovanni Antonio Monteverdi was an Italian choirmaster.
Sentence 1:  A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance and the Baroque periods of music history.
  Fact 1.0: Monteverdi composed both secular and sacred music.
  Fact 1.1: Monteverdi was a pioneer in the development of opera.
  Fact 1.2: Monteverdi is considered a crucial transitional figure between the Renaissance and the Baroque periods of music history.
Sentence 2: Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.
  Fact 2.0: Georges Bizet was born on 25 October 1838.
  Fact 2.1: Georges Bizet died on 3 June 1875.
  Fact 2.2: Georges Bizet was registered at birth as Alexandre César Léopold Bizet.
  Fact 2.3: Georges Bizet was a French composer.
  Fact 2.4: Georges Bizet was a composer of the romantic era.
Sentence 3:  Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the most popular and frequently performed works in the entire opera repertoire.
  Fact 3.0: Georges Bizet is best known for his operas.
  Fact 3.1: Bizet's career was cut short by his early death.
  Fact 3.2: Bizet achieved few successes before his final work.
  Fact 3.3: Bizet's final work is 'Carmen'.
  Fact 3.4: Carmen has become one of the most popular works in the entire opera repertoire.
  Fact 3.5: Carmen is one of the most frequently performed works in the entire opera repertoire.

Retrieved items:
<item 1>
Symphony in C (Bizet)
The Symphony in C is an early work by the French composer Georges Bizet.

 According to "Grove's Dictionary", the symphony "reveals an extraordinarily accomplished talent for a 17-year-old student, in melodic invention, thematic
handling and orchestration."

 Bizet started work on the symphony in C major on 29 October 1855, four days after turning 17, and finished it roughly a month later.

 It was written while he was studying at the Paris Conservatoire under the composer Charles Gounod, and was evidently a student assignment.

 Bizet showed no apparent interest in having it performed or published, and the piece was never played in his lifetime.

 He used certain material from the symphony in later works, however.

 There is no mention of the work in Bizet's letters, and it was unknown to his earlier biographers.

 His widow, Geneviève Halévy (1849–1926), gave the manuscript to Reynaldo Hahn, who left it along with other papers to the archives of the conservatory library,
where it was found in 1933 by Jean Chantavoine.
</item 1>

<item 2>
Georges Bizet
Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.

 Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the
most popular and frequently performed works in the entire opera repertoire.
</item 2>

<item 3>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 3>

<item 4>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 4>

<item 5>
John Whenham
John Whenham is an English musicologist and academic who specializes in early Italian baroque music.

 He earned both a Bachelor of Music and a Master of Music from the University of Nottingham, and a Doctor of Philosophy from the University of Oxford.

 He is a leading expert on the life and works of Claudio Monteverdi, and is the author of the books "Duet and Dialogue in the Age of Monteverdi" (Ann Arbor,
Michigan: University Microfilms International, 1982) "Monteverdi, 'Orfeo' " (London: Cambridge University Press, 1986), "Monteverdi, Vespers (1610)" (Cambridge
University Press, 1997), and "The Cambridge Companion to Monteverdi" (with Richard Wistreich, Cambridge University Press, 2007).

 For five years he was co-editor of the journal "Music & Letters".

 He currently serves on the board of the Birmingham Early Music Festival and is head of the music history department at the University of Birmingham.
</item 5>

<item 6>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 6>

<item 7>
Roma Symphony (Bizet)
The Symphony in C "Roma" is the second of Georges Bizet's symphonies.

 Unlike his first symphony, also in C major, which was written quickly at the age of 17, "Roma" was written over an eleven-year span, between the ages of 22 and
33 (he died at age 36).

 Bizet was never fully satisfied with it, subjecting it to a number of revisions, but died before finishing his definitive version.

 All four movements were performed in his lifetime, but never all on the same occasion.

 The full symphony in its latest revision was premiered in 1875, after his death.

 It is perhaps because of Bizet's dissatisfaction that the work is often said to be "unfinished".

 However, in the form in which it exists today, it is a complete work and is fully scored.

 It has been recorded a number of times but is not often heard on the concert platform.
</item 7>

<item 8>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 8>

<item 9>
L'Orfeo discography
The 1969 recording by Nicholas Harnoncourt and the Vienna Concentus Musicus, using Harnoncourt's edition based on period instruments, was praised for "making
Monteverdi's music sound something like the way he imagined".

 In 1981 Siegfried Heinrich, with the Early Music Studio of the Hesse Chamber Orchestra, recorded a version which re-created the original Striggio libretto
ending, adding music from Monteverdi's 1616 ballet "Tirsi e Clori" for the Bacchante scenes.

 Among more recent recordings, that of Emmanuelle Haïm has been praised for its dramatic effect.

 The 21st century has seen the issue of an increasing number of recordings on DVD.
</item 9>

<item 10>
L'Orfeo discography
These lists show the audio and visual recordings of the opera "L'Orfeo" by Claudio Monteverdi.

 The opera was first performed in Mantua in 1607, at the court of Duke Vincenzo Gonzaga, and is one of the earliest of all operas.

 The first recording of "L'Orfeo" was issued in 1939, a freely adapted version of Monteverdi's music edited by Giacomo Benvenuti, given by the orchestra of La
Scala Milan conducted by Ferrucio Calusio.

 In 1949 the Berlin Radio Orchestra under Helmut Koch recorded the complete opera, on long-playing records (LPs).

 The advent of LP recordings was, as Harold Schonberg later wrote, an important factor in the postwar revival of interest in Renaissance and Baroque music, and
from the mid-1950s recordings of "L'Orfeo" have been issued on many labels.

 Koch's landmark version was reissued in 1962, when it was compared unfavourably with others that had by then been issued.
</item 10>
=== END ITEM id=594ea96b1b810d72 ===

=== ITEM id=087f7ee4b7ffc019 ===
Gold sentences and facts:
Sentence 0: Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.
  Fact 0.0: Ravi Sethi was born in 1947.
  Fact 0.1: Ravi Sethi is an Indian computer scientist.
  Fact 0.2: Ravi Sethi retired from Bell Labs.
  Fact 0.3: Ravi Sethi is president of Avaya Labs Research.
Sentence 1: Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company, owned by Finnish company Nokia.
  Fact 1.0: Nokia Bell Labs was formerly named AT&T Bell Laboratories.
  Fact 1.1: Nokia Bell Labs was formerly named Bell Telephone Laboratories.
  Fact 1.2: Nokia Bell Labs was formerly named Bell Labs.
  Fact 1.3: Nokia Bell Labs is an American research and scientific development company.
  Fact 1.4: Nokia Bell Labs is owned by the Finnish company Nokia.
Sentence 2:  Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
  Fact 2.0: The headquarters of Nokia Bell Labs are located in Murray Hill, New Jersey.
  Fact 2.1: Nokia Bell Labs has other laboratories around the rest of the United States.
  Fact 2.2: Nokia Bell Labs has laboratories in other countries.

Retrieved items:
<item 1>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 1>

<item 2>
Center for Climate Systems Research (CCSR)
The Center for Climate Systems Research is a key Earth Institute center with over 25 scientists and staff researching issues involving the interplay between
Earth's climate and society.

 The center's motto is "Columbia's Gateway to NASA and Beyond", as it has a special relationship with the NASA Goddard Institute for Space Studies (GISS).

 It is a "Gateway to NASA", because the center is co-located with GISS in Columbia University's Armstrong Hall above Tom's Restaurant in New York City.

 In this historic location, Columbia and NASA scientists work together closely to understand the Earth's climate and how changes can impact society both in the
United States and around the world.

 Importantly, CCSR scientists also work with many other partners – governmental, private sector, and non-governmental organizations – to answer these
challenging scientific and socioeconomic questions (i.e., "and Beyond").

 The director of CCSR is Dr. Michael J. Puma.
</item 2>

<item 3>
Bell Labs
Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company,
owned by Finnish company Nokia.

 Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
</item 3>

<item 4>
Pharma Medica
Pharma Medica Research Inc. is an American research and development company that works with pharmaceutical companies and performs clinical trials in the
pharmaceutical and biotechnology industry.

 The company is based in Toronto, Canada with headquarters in Mississauga, Ontario.

 Pharma Medica has locations in Scarborough, Ontario and St. Charles, Missouri.
</item 4>

<item 5>
Umesh Varshney
Umesh Varshney (born 1957) is an Indian molecular biologist, academician and the head of the "Prof.Umesh Varshney's Lab" at the Indian Institute of Science,
Bengaluru.

 He is a J. C. Bose National Fellow of the Department of Science and Technology and is known for his studies on protein synthesis and DNA repair in "Escherichia
coli" and Mycobacterium tuberculosis.

 An elected fellow of the Indian Academy of Sciences, Indian National Science Academy and the National Academy of Sciences (India), he is also a recipient of
the National Bioscience Award for Career Development of the Government of India.

 The Council of Scientific and Industrial Research, the apex agency of the Government of India for scientific research, awarded him the Shanti Swarup Bhatnagar
Prize for Science and Technology, one of the highest Indian science awards, in 2001, and then in 2014 with the G. N. Ramachandran Gold Medal for Excellence in
Biological Sciences & Technology for his contributions to biological sciences.
</item 5>

<item 6>
Ravi Sethi
Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.

 He also serves as a member of the National Science Foundation's Computer and Information Science and Engineering (CISE) Advisory Committee.

 He is best known as one of three authors of the classic computer science textbook "", also known as the "Dragon Book".
</item 6>

<item 7>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 7>

<item 8>
Compilers: Principles, Techniques, and Tools
Compilers: Principles, Techniques, and Tools is a computer science textbook by Alfred V. Aho, Monica S. Lam, Ravi Sethi, and Jeffrey D. Ullman about compiler
construction.

 First published in 1986, it is widely regarded as the classic definitive compiler technology text.
</item 8>

<item 9>
Sethi–Ullman algorithm
In computer science, the Sethi–Ullman algorithm is an algorithm named after Ravi Sethi and Jeffrey D. Ullman, its inventors, for translating abstract syntax
trees into machine code that uses as few registers as possible.
</item 9>

<item 10>
Kenneth Hess
Kenneth Lafferty Hess (born January 22, 1953) is an engineer, author, entrepreneur, and philanthropist.

 Hess is founder and president of Science Buddies, a non-profit organization dedicated to furthering science literacy through the creation of free resources and
services for K-12 students, teachers, and families.

 He holds one of the first software patents ever granted and has designed and/or developed dozens of commercial software, content, and Internet products,
including Family Tree Maker, one of the all-time best-selling home software programs.

 Among his awards are a "PC Magazine" Editor's Choice, "PC Magazine" Top 100 Web Site, a Software Publishing Association Codie, and a "Science" Prize for Online
Resources in Education (SPORE).
</item 10>
=== END ITEM id=087f7ee4b7ffc019 ===
