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

This file has 12 items.

=== ITEM id=36f61da6f2d09103 ===
Gold sentences and facts:
Sentence 0: Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.
  Fact 0.0: Heavenly Creatures is a 1994 New Zealand psychological drama.
  Fact 0.1: Heavenly Creatures was directed by Peter Jackson.
  Fact 0.2: Peter Jackson co-wrote the screenplay with his partner, Fran Walsh.
  Fact 0.3: Heavenly Creatures is about the notorious 1954 Parker-Hulme murder case.
  Fact 0.4: The Parker-Hulme murder case took place in Christchurch, New Zealand.
Sentence 1:  The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.
  Fact 1.0: The main premise of Heavenly Creatures deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme.
  Fact 1.1: Pauline Parker and Juliet Hulme murder Pauline Parker's mother.
Sentence 2: Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk series.
  Fact 2.0: Anne Perry was born on 28 October 1938 as Juliet Marion Hulme.
  Fact 2.1: Anne Perry is an English author of historical detective fiction.
  Fact 2.2: Anne Perry is best known for her Thomas Pitt and William Monk series.

Retrieved items:
<item 1>
Jane Henschel
Jane Henschel (born 2 March 1952) is an American operatic mezzo soprano.

 Henschel, who was born in Wisconsin, studied at the University of Southern California, and then pursued further studies in Germany, where she has made her
home.

 Her numerous opera appearances include Baba the Turk in Igor Stravinsky's "The Rake’s Progress" with Glyndebourne Festival Opera, the Saito Kinen Festival
Matsumoto, and the Salzburg festival; Brangäne in Richard Wagner’s "Die Walküre" with Paris Opéra and the Los Angeles Opera; the Principessa in Giacomo
Puccini’s "Suor Angelica" with conductor Riccardo Chailly and the Royal Concertgebouw Orchestra; Blanche de la Force in Francis Poulenc’s "Dialogues des
Carmélites" in Amsterdam; Kostelnicka Buryjovka in Leoš Janáček’s "Jenůfa" under Seiji Ozawa in Japan; and the Kabanicka in Janáček’s "Katya Kabanova" at the
Salzburg Festival among others.
</item 1>

<item 2>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 2>

<item 3>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 3>

<item 4>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 4>

<item 5>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 5>

<item 6>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 6>

<item 7>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 7>

<item 8>
Heavenly Creatures
Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the
notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.

 The film features Melanie Lynskey and Kate Winslet in their screen debuts with supporting roles by Sarah Peirse, Diana Kent, Clive Merrison, and Simon
O'Connor.

 The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.

 The events of the film cover the period from their meeting in 1952 to the murder in 1954.
</item 8>

<item 9>
Peter Elliott (actor)
Peter Elliott is a leading New Zealand actor.

 He has appeared in numerous television shows including "Shortland Street", "Gloss" and .

 He has also appeared in several movies including "Heavenly Creatures".

 Peter has a daughter Lucy Elliott who is an actress, playing character Dayna Jenkins on "Shortland Street" in 2013.
</item 9>

<item 10>
Parker–Hulme murder case
The Parker–Hulme murder case began in the city of Christchurch, New Zealand, on 22 June 1954, when Honora Rieper (also known as Honora Parker, her legal name)
was killed by her teenaged daughter, Pauline Parker, and Pauline's close friend Juliet Hulme (later known as Anne Perry).

 Parker was 16 at the time, while Hulme was 15.
</item 10>
=== END ITEM id=36f61da6f2d09103 ===

=== ITEM id=adce55ef25d85f9f ===
Gold sentences and facts:
Sentence 0:  Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
  Fact 0.0: The unnamed person in this passage has other voice-over roles
  Fact 0.1: The unnamed person voiced Dr. Drakken in Kim Possible
  Fact 0.2: The unnamed person voiced Motor Ed in Kim Possible
  Fact 0.3: The unnamed person voiced Brother Blood in Teen Titans
  Fact 0.4: The unnamed person voiced Aquaman in an unnamed work
  Fact 0.5: The unnamed person voiced Rico in The Penguins of Madagascar
  Fact 0.6: The unnamed person voiced Ogre in an unnamed work
  Fact 0.7: The unnamed person voiced Niblet in Pound Puppies
  Fact 0.8: The unnamed person voiced the Scotsman in Samurai Jack
  Fact 0.9: The unnamed person voiced Shnitzel in Chowder
Sentence 1: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 1.0: Chowder is an American animated television series
  Fact 1.1: Chowder was created by C. H. Greenblatt
  Fact 1.2: Chowder was created for Cartoon Network

Retrieved items:
<item 1>
Band Geeks
"Band Geeks" is the second part of the 15th episode of the second season, and the second half of the 35th episode overall, of the American animated television
program "SpongeBob SquarePants".

 It originally aired on Nickelodeon in the United States on September 7, 2001.

 It was written by C. H. Greenblatt, Aaron Springer, and Merriwether Williams, and the animation was directed by Frank Weiss.

 Springer served as storyboard director, and Greenblatt served as storyboard artist.

 The song "Sweet Victory" by David Glen Eisley was featured in the episode and was later released on the album "" in 2005.
</item 1>

<item 2>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 2>

<item 3>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 3>

<item 4>
List of Adventure Time episodes
"Adventure Time" is an American animated television series created by Pendleton Ward for Cartoon Network.

 The series follows the adventures of Finn (voiced by Jeremy Shada), a human boy, and his best friend and adoptive brother Jake (voiced by John DiMaggio), a dog
with magical powers to change shape and grow and shrink at will.

 Finn and Jake live in the post-apocalyptic Land of Ooo.

 Throughout the series, they interact with the show's other main characters: Princess Bubblegum (voiced by Hynden Walch), the sovereign of the Candy Kingdom;
the Ice King (voiced by Tom Kenny), a demented but largely misunderstood ice wizard; Marceline the Vampire Queen (voiced by Olivia Olson), a thousand-year-old
vampire and rock music enthusiast; Lumpy Space Princess (voiced by Ward), a melodramatic and immature princess made out of "irradiated stardust"; BMO (voiced by
Niki Yang), a sentient video game console-shaped robot that lives with Finn and Jake; and Flame Princess (voiced by Jessica DiCicco), a flame elemental and
ruler of the Fire Kingdom.
</item 4>

<item 5>
Harvey Beaks
Harvey Beaks is an American animated television series created by C.H. Greenblatt for Nickelodeon.

 The series premiered on March 28, 2015 following the 2015 Kids' Choice Awards.
</item 5>

<item 6>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 6>

<item 7>
Adventure Time (pilot)
"Adventure Time" is an animated short created by Pendleton Ward, as well as the pilot to the Cartoon Network series of the same name.

 The short follows the adventures of Pen (voiced by Zack Shada), a human boy, and his best friend Jake (voiced by John DiMaggio), a dog with magical powers to
change shape and grow and shrink at will.

 In this episode, Pen and Jake have to rescue Princess Bubblegum (voiced by Paige Moss) from the antagonistic Ice King (voiced by John Kassir).
</item 7>

<item 8>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 8>

<item 9>
Adventure Time
Adventure Time is an American animated television series created by Pendleton Ward for Cartoon Network.

 It follows the adventures of a boy named Finn (voiced by Jeremy Shada) and his best friend and adoptive brother Jake (John DiMaggio)— a dog with the magical
power to change shape and size at will.

 Finn and Jake live in the post-apocalyptic Land of Ooo, where they interact with Princess Bubblegum (Hynden Walch), the Ice King (Tom Kenny), Marceline the
Vampire Queen (Olivia Olson), BMO (Niki Yang), and others.

 The series is based on a 2007 short produced for Nicktoons and Frederator Studios' animation incubator series "Random!

 Cartoons".

 After the short became a viral hit on the Internet, Cartoon Network commissioned a full-length series, which previewed on March 11, 2010, and officially
premiered on April 5, 2010.
</item 9>

<item 10>
Elements (miniseries)
Elements is an American animated miniseries based on the show "Adventure Time" by Pendleton Ward.

 It aired as part of the show's ninth season on Cartoon Network from April 24 to April 27, 2017.

 "Adventure Time" follows the adventures of Finn (voiced by Jeremy Shada), a human boy, and his best friend and adoptive brother Jake (voiced by John DiMaggio),
a dog with magical powers to change shape, grow and shrink at will.

 In this limited event series, which itself follows the events of the "Islands" miniseries, Finn, Jake, and BMO return home to discover that Ooo has been turned
into a veritable dystopia thanks to extreme elemental magic.

 Finn and Jake team up with Ice King (voiced by Tom Kenny), Betty (voiced by Felicia Day), and Lumpy Space Princess (voiced by Ward) to set things straight.
</item 10>
=== END ITEM id=adce55ef25d85f9f ===

=== ITEM id=a96524e5a29595c4 ===
Gold sentences and facts:
Sentence 0: The University of Delaware (colloquially "UD") is the largest university in Delaware.
  Fact 0.0: The University of Delaware is colloquially called 'UD'.
  Fact 0.1: The University of Delaware is the largest university in Delaware.
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen's Hungarian name is 'Debreceni Egyetem'.
  Fact 1.1: The University of Debrecen is a university.
  Fact 1.2: The University of Debrecen is located in Debrecen, Hungary.

Retrieved items:
<item 1>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 1>

<item 2>
St. John's College, Belize
St. John's College has three divisions, and a number of central academic centres and activities.

 Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college
levels.

 St. John's College is a Roman Catholic institution in the Jesuit tradition, one of the oldest, largest, and most diverse educational institutions in Belize,
founded by the Jesuits in 1887.
</item 2>

<item 3>
Rutgers University–Camden
Rutgers University–Camden is one of three regional campuses of Rutgers University, New Jersey's public research university.

 It is located in Camden, New Jersey, United States.

 Founded in the 1920s, Rutgers–Camden began as an amalgam of the South Jersey Law School and the College of South Jersey.

 It is the southernmost of the three regional campuses of Rutgers—the others being located in New Brunswick and Newark.

 The city of Camden is located on the Delaware River, east of Philadelphia.
</item 3>

<item 4>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 4>

<item 5>
University of Delaware Botanic Gardens
The University of Delaware Botanic Gardens are botanical gardens and an arboretum located on the campus of the University of Delaware, in Newark, Delaware,
United States.

 The gardens are open to the public without charge.
</item 5>

<item 6>
Ellendale, Delaware
Ellendale is a town in Sussex County, Delaware, United States.

 The population was 381 at the 2010 census, an increase of 16.5% since 2000.

 It is part of the Salisbury, Maryland-Delaware Metropolitan Statistical Area.

 Ellendale is the "Gateway to Delaware's Resort Beaches" because it is the town located on U.S. Highway 113, the resort area's westernmost border, and Delaware
Route 16, the resort area's northernmost border with the eastern border being the Delaware Bay and Atlantic Ocean and the southern border being the state line
with Maryland.

 Ellendale is home to the Philadelphia Bible College and the Harbor Christian Academy.
</item 6>

<item 7>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 7>

<item 8>
College Park, Delaware
College Park is an unincorporated community in New Castle County, Delaware, United States.

 College Park is located along Delaware Route 896 southwest of Newark.
</item 8>

<item 9>
Delaware–William &amp; Mary football rivalry
The Delaware–William & Mary football rivalry between the Delaware Fightin' Blue Hens and the William & Mary Tribe is a match-up between two public universities,
the University of Delaware and the College of William and Mary, that are also members of the Colonial Athletic Association.

 Both schools have academic reputations that have labeled them as Public Ivies.

 Both schools are also Colonial Colleges having been founded before the United States became independent in 1776; the College of William and Mary was founded in
1693 and the University of Delaware's predecessor school was founded in 1743.
</item 9>

<item 10>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 10>
=== END ITEM id=a96524e5a29595c4 ===

=== ITEM id=0048cadaa2d348ea ===
Gold sentences and facts:
Sentence 0: Akademisches Kunstmuseum (English:Academic Art Museum) is an art museum in Bonn, Germany.
  Fact 0.0: Akademisches Kunstmuseum is also known in English as the Academic Art Museum.
  Fact 0.1: Akademisches Kunstmuseum is an art museum.
  Fact 0.2: Akademisches Kunstmuseum is located in Bonn, Germany.
Sentence 1: The Federal City of Bonn (] ) is a city on the banks of the Rhine in the German state of North Rhine-Westphalia, with a population of over 300,000.
  Fact 1.0: The Federal City of Bonn is a city.
  Fact 1.1: The Federal City of Bonn is on the banks of the Rhine.
  Fact 1.2: The Federal City of Bonn is in the German state of North Rhine-Westphalia.
  Fact 1.3: The Federal City of Bonn has a population of over 300,000.

Retrieved items:
<item 1>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 1>

<item 2>
Akademisches Kunstmuseum
Akademisches Kunstmuseum (English:Academic Art Museum) is an art museum in Bonn, Germany.

 It is the oldest museum in Bonn and houses the antique collection of the University of Bonn with more than 500 antique statues and reliefs, and over 2,000
originals.

 It is located in a neoclassical building at the southern end of the Hofgarten, near the Electoral Palace.
</item 2>

<item 3>
National Museum, Kraków
The National Museum in Kraków (Polish: "Muzeum Narodowe w Krakowie" ), popularly abbreviated as MNK, established in 1879, is the main branch of Poland's
National Museum, which has several independent branches with permanent collections around the country.

 The Museum consists of 21 departments which are divided by art period; 11 galleries, 2 libraries, and 12 conservation workshops.

 It holds some 780,000 art objects, spanning from classical archeology to modern art, with special focus on Polish painting.

<ref name="culture.pl/muzea"> </ref><ref name="muzeum/Zbiory"> </ref>
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
Bonn
The Federal City of Bonn (] ) is a city on the banks of the Rhine in the German state of North Rhine-Westphalia, with a population of over 300,000.

 About 24 km south-southeast of Cologne, Bonn is in the southernmost part of the Rhine-Ruhr region, Germany's largest metropolitan area, with over 11 million
inhabitants.
</item 5>

<item 6>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 6>

<item 7>
Croatian Natural History Museum
The Croatian Natural History Museum (Croatian: "Hrvatski prirodoslovni muzej" ) is the oldest and biggest natural history museum and the main body for natural
history research, preservation and collection in Croatia.

 Located on Dimitrije Demeter Street in Gornji Grad, one of the oldest neighbourhoods of the Croatian capital Zagreb, it owns one of the biggest museum
collections in Croatia, with over 2 million artefacts, including over 1.1 million animal specimens.

 It was founded in 1846 as the "National Museum".

 The National Museum was later split up into five museums, three of which were in 1986 merged as departments of the newly named Croatian Natural History Museum.

 The museum contains a large scientific library open to the public, and publishes the first Croatian natural history scientific journal, "Natura Croatica".
</item 7>

<item 8>
Palace of Music (Miskolc)
The Palace of Music ("Zenepalota") is a building in Bartók square, Miskolc, Hungary.

 It is the building of the Béla Bartók Secondary School and the Béla Bartók Music Institute (a faculty of the University of Miskolc.)

 The Palace was designed by Gyula Waelder in Neo-baroque style and was built between 1926 and 1927.

 The construction was financed from USA loans, just like that of the Hotel Palace in Lillafüred and the Market Hall on Búza tér.
</item 8>

<item 9>
National Museum of Poland
"National Museum of Poland" is the common name for several of the country's largest and most notable museums. Poland's National Museum comprises several
independent branches, each operating a number of smaller museums.

 The main branch is the National Museum in Kraków (Polish: "Muzeum Narodowe w Krakowie" ), established in 1879 with permanent collections consisting of several
hundred thousand items – kept in big part at the Main Building (along the 3 Maja St.), but also in the eight of its divisions around the city.
</item 9>

<item 10>
List of Kerala cities by population
The article lists major cities in Kerala based on the total population in city corporations or municipalities and their outgrowths.

 This list is based on the latest census data.

 Population of cities after major mergers post to latest census are also included.
</item 10>
=== END ITEM id=0048cadaa2d348ea ===

=== ITEM id=063e5ab375f9f03b ===
Gold sentences and facts:
Sentence 0: Samsung Galaxy S7 and Samsung Galaxy S7 Edge are Android smartphones manufactured and marketed by Samsung Electronics.
  Fact 0.0: Samsung Galaxy S7 is an Android smartphone.
  Fact 0.1: Samsung Galaxy S7 Edge is an Android smartphone.
  Fact 0.2: Samsung Galaxy S7 and S7 Edge are manufactured by Samsung Electronics.
  Fact 0.3: Samsung Galaxy S7 and S7 Edge are marketed by Samsung Electronics.
Sentence 1:  The S7 series serves as the successor to the Galaxy S6, S6 Edge and S6 Edge+ released in 2015.
  Fact 1.0: The S7 series is the successor to the Galaxy S6.
  Fact 1.1: The S7 series is the successor to the Galaxy S6 Edge.
  Fact 1.2: The S7 series is the successor to the Galaxy S6 Edge+.
  Fact 1.3: The Galaxy S6, S6 Edge and S6 Edge+ were released in 2015.
Sentence 2: Samsung Galaxy S6, Samsung Galaxy S6 Edge, and Samsung Galaxy S6 Edge+ are Android smartphones manufactured and marketed by Samsung Electronics.
  Fact 2.0: Samsung Galaxy S6 is an Android smartphone.
  Fact 2.1: Samsung Galaxy S6 Edge is an Android smartphone.
  Fact 2.2: Samsung Galaxy S6 Edge+ is an Android smartphone.
  Fact 2.3: Samsung Galaxy S6, S6 Edge and S6 Edge+ are manufactured by Samsung Electronics.
  Fact 2.4: Samsung Galaxy S6, S6 Edge and S6 Edge+ are marketed by Samsung Electronics.
Sentence 3:  The S6 line serves as a successor to the Galaxy S5.
  Fact 3.0: The S6 line is a successor to the Galaxy S5.
Sentence 4:  The S6 and S6 Edge smartphones were officially unveiled in the first "Samsung Unpacked 2015" event at Mobile World Congress on 1 March 2015, while the bigger S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5 in the second "Samsung Unpacked 2015" event at New York on 13 August 2015.
  Fact 4.0: The S6 was officially unveiled in the first Samsung Unpacked 2015 event.
  Fact 4.1: The S6 Edge was officially unveiled in the first Samsung Unpacked 2015 event.
  Fact 4.2: The first Samsung Unpacked 2015 event was held at Mobile World Congress on 1 March 2015.
  Fact 4.3: The S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5.
  Fact 4.4: The S6 Edge+ was officially unveiled in the second Samsung Unpacked 2015 event.
  Fact 4.5: The second Samsung Unpacked 2015 event was held in New York on 13 August 2015.

Retrieved items:
<item 1>
Samsung SGH-D500
The Samsung SGH-D500 is a slider-style mobile phone created by Samsung.

 It was announced in Q4 2004 as a replacement to the popular Samsung E800.

 It has a 1.3 megapixel digital camera with an integrated LED flash.

 The phone can be purchased in various colors such as black, grey/silver, silver/white, silver/blue, black/blue, and black/silver.

 The D500 won the prestigious award "the worlds best cellphone" at the 3GSM world congress in Cannes, 2005.

 The SGH-D500 was popular with consumers because of the large screen-size to front-surface ratio, the smooth button and sliding action, powerful flash, and good
video/photograph/sound quality.

 The D500 was regarded as compact and as having a good feature set for its time.

 Samsung experienced remarkable success with the introduction of the D500 and has therefore introduced several other slider-style mobile phones in recent years,
some fairly successful, others not so much.

 This phone helped to popularise the 'active' sliding phone concept across all brands.

 The SGH-D500 has been succeeded by the Samsung SGH-D600.
</item 1>

<item 2>
Samsung SGH-E900
The Samsung E900, introduced in 2006, is a high-end mobile phone and is derived from Samsung's D500-D600-D800 series of slide phones.

 It is of a slide-up design and has touch-sensitive keys, similar to the LG Chocolate (KG800).

 It was one of the most popular phones at the time but was discontinued late 2007 when buyers then turned their attention to the Samsung D900i, followed by the
Samsung U600, which was the slimmest phone by Samsung with a thickness of 1.09 cm until the E840 was released.
</item 2>

<item 3>
Samsung Galaxy S7
Samsung Galaxy S7 and Samsung Galaxy S7 Edge are Android smartphones manufactured and marketed by Samsung Electronics.

 The S7 series serves as the successor to the Galaxy S6, S6 Edge and S6 Edge+ released in 2015.

 The phones were officially unveiled on 21 February 2016, during a Samsung press conference at Mobile World Congress, with a European and North American release
on 11 March 2016.
</item 3>

<item 4>
Samsung Ativ S
The Samsung ATIV S is a touchscreen, slate smartphone manufactured by Samsung Electronics running the Windows Phone 8 operating system, upgradeable to Windows
Phone 8.1.

 The ATIV S is Samsung's first Windows Phone 8 device, and one of the first devices under its ATIV series of Microsoft Windows-based products.

 It was shown at the IFA 2012 in Berlin and was the first Windows Phone 8 device to be officially unveiled, ahead of the Nokia Lumia 920.
</item 4>

<item 5>
Samsung SGH-T919
The Samsung Behold SGH-T919 is a touch-screen, 3G candybar-style mobile phone introduced by Samsung late in 2008.

 The Behold is one of the first Samsung mobile phones released to have a touch-screen along with the Samsung Omnia, the Samsung Instinct, the Samsung Eternity,
and the Samsung Tocco.

 The Behold is the "American" release of Samsung Tocco, which is only sold in Europe, Asia and Australia so the features and design of both of these phones are
similar.

 Some of the Behold's biggest competitors were Apple's iPhone, T-Mobile G1 (HTC Dream), LG Dare, Samsung Instinct, and Samsung Eternity.

 It was only available only to customers of T-Mobile.
</item 5>

<item 6>
Samsung SGH-D500
The SGH-D500 has been succeeded by the Samsung SGH-D600.

 There is a very similar variant handset, the SGH-D500E.
</item 6>

<item 7>
Samsung Galaxy S Duos 2
Samsung Galaxy S Duos 2 is a version of Samsung Galaxy Trend Plus with dual SIM Android smartphone, produced and marketed by Samsung Electronics, which serves
as an immediate successor to the original Galaxy S Duos, which is made in second quarter of 2012.

 Unveiled on 30 November 2013 at some Asian markets, it was initially released on December 5, 2013 in several countries.

 In contrast with other dual sim Samsung models, this phone is a part of the high-end "S" series, this is why it is marketed as a part of the "Galaxy S" family.

 Despite being called as a direct successor to the original S Duos phone, external and physical design of the model is identical to the original model, placing
emphasis on internal upgrades such as upgraded processor and updated Operating System software.
</item 7>

<item 8>
Samsung Galaxy S6
Samsung Galaxy S6, Samsung Galaxy S6 Edge, and Samsung Galaxy S6 Edge+ are Android smartphones manufactured and marketed by Samsung Electronics.

 The S6 line serves as a successor to the Galaxy S5.

 The S6 and S6 Edge smartphones were officially unveiled in the first "Samsung Unpacked 2015" event at Mobile World Congress on 1 March 2015, while the bigger
S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5 in the second "Samsung Unpacked 2015" event at New York on 13 August 2015.

 Alongside the S6, Samsung also unveiled the S6 Edge (and later on the bigger S6 Edge+), a variant whose screen is wrapped along the sides of the device; the
curvature is usable for several additional features.

 The Galaxy S6 and S6 Edge were released on 10 April 2015 in 20 countries while the S6 Edge+ was released on 21 August 2015 in 20 countries.
</item 8>

<item 9>
Samsung Galaxy Gear
The Samsung Galaxy Gear is a smartwatch produced by Samsung Electronics in the Samsung Gear family of devices.

 Unveiled during a "Samsung Unpacked" event in Berlin, Germany on September 4, 2013, the device serves as a companion for all Samsung Galaxy smartphones and
tablets which run Android 4.3 or newer; it was released on September 25, 2013.

 Originally released as an Android-based device, Samsung replaced the operating system with Tizen through the May 2014 software update.
</item 9>

<item 10>
Samsung REX
The Samsung REX series is a range of Samsung smartphones and feature phones, ranging from low to mid-end phones, running a Java-based operating system with
Samsung's TouchWiz user interface.

 It is currently one of the Samsung line of devices that are aimed at the budget phone market alongside lower end Samsung Galaxy devices.

 Its main competition includes the Nokia Asha range and Firefox OS.
</item 10>
=== END ITEM id=063e5ab375f9f03b ===

=== ITEM id=47ef14c420ba22a4 ===
Gold sentences and facts:
Sentence 0:  Her only novel, "The Bone People", won the Booker Prize in 1985.
  Fact 0.0: Her only novel is 'The Bone People'.
  Fact 0.1: 'The Bone People' won the Booker Prize.
  Fact 0.2: 'The Bone People' won the Booker Prize in 1985.
Sentence 1:  His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science fiction's top U.S. prizes in consecutive years.
  Fact 1.0: Card's novel 'Ender's Game' was published in 1985.
  Fact 1.1: 'Speaker for the Dead' is the sequel to 'Ender's Game'.
  Fact 1.2: 'Speaker for the Dead' was published in 1986.
  Fact 1.3: 'Ender's Game' won the Hugo Award.
  Fact 1.4: 'Ender's Game' won the Nebula Award.
  Fact 1.5: 'Speaker for the Dead' won the Hugo Award.
  Fact 1.6: 'Speaker for the Dead' won the Nebula Award.
  Fact 1.7: Card is the only author to win both science fiction's top U.S. prizes in consecutive years.

Retrieved items:
<item 1>
Ender in Exile
Ender in Exile is a science fiction novel by Orson Scott Card, part of the "Ender's Game" series, published on November 11, 2008.

 It takes place between the two award-winning novels: "Ender's Game" and "Speaker for the Dead".

 It could also be considered a parallel novel to the first three sequels in the Shadow Saga, since the entirety of this "trilogy" takes place in the span of
"Ender in Exile".

 The novel concludes a dangling story line of the Shadow Saga, while it makes several references to events that take place during the Shadow Saga.

 From yet another perspective, the novel expands (or "replaces") the last chapter of the original novel "Ender's Game".

 On the one hand, it fills the gap right before the last chapter, and on the other hand, it fills the gap between the last chapter and the original (first)
sequel (both named "Speaker for the Dead").
</item 1>

<item 2>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 2>

<item 3>
Orson Scott Card bibliography
This is a list of the works of Orson Scott Card.

 This list does not include criticisms, reviews, or related material written by Card.

 Orson Scott Card is the author of The Ender saga and Homecoming Saga among many other works.
</item 3>

<item 4>
Shadow of the Hegemon
Shadow of the Hegemon (2001) is the second novel in the "Ender's Shadow" series (often called the Bean Quartet) by Orson Scott Card.

 It is also the sixth novel in the "Ender's Game" series.

 It is told mostly from the point of view of Bean, a largely peripheral character in the original novel "Ender's Game" but the central protagonist of the
parallel narrative "Ender's Shadow."

 "Shadow of the Hegemon" was nominated for a Locus Award in 2002.
</item 4>

<item 5>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 5>

<item 6>
Ender in Exile
"Ender in Exile" begins one year after Ender has won the bugger war, and begins with the short story "Ender's Homecoming" from Card's webzine Intergalactic
Medicine Show.

 Other short stories that were published elsewhere are included as chapters of the novel.
</item 6>

<item 7>
A War of Gifts: An Ender Story
A War of Gifts: An Ender Story (2007) is a science fiction novel by Orson Scott Card.

 This book is set in Card's Ender's Game series and takes place during Ender Wiggin's time at Battle School as described in Card's novels "Ender's Game" and
"Ender's Shadow".
</item 7>

<item 8>
Ender's Game (short story)
"Ender's Game" is a story by Orson Scott Card.

 It first appeared in the August 1977 issue of "Analog" magazine and was later expanded into the novel "Ender's Game".

 Although the foundation of the Ender's Game series, the short story is not properly part of the "Ender's Game" universe, as there are many discrepancies in
continuity.
</item 8>

<item 9>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 9>

<item 10>
Ender Wiggin
Andrew "Ender" Wiggin is a fictional character from Orson Scott Card's 1985 science fiction novel "Ender's Game" and its sequels ("Speaker for the Dead",
"Xenocide", "Children of the Mind", "Ender in Exile"), as well as in the first part of the spin-off series, "Ender's Shadow".

 The book series itself is an expansion, with some changes to detail, of Card's 1977 short story "Ender's Game."
</item 10>
=== END ITEM id=47ef14c420ba22a4 ===

=== ITEM id=1d958d39819b278c ===
Gold sentences and facts:
Sentence 0: Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at Naval Air Station Whidbey Island, Washington.
  Fact 0.0: Electronic Attack Squadron 138 (VAQ-138) is also known as the "Yellow Jackets".
  Fact 0.1: Electronic Attack Squadron 138 is an expeditionary EA-18G Growler squadron of the United States Navy.
  Fact 0.2: Electronic Attack Squadron 138 is based at Naval Air Station Whidbey Island, Washington.
Sentence 1: The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.
  Fact 1.0: The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft.
  Fact 1.1: The Boeing EA-18G Growler is a specialized version of the two-seat F/A-18F Super Hornet.

Retrieved items:
<item 1>
VFA-154
Strike Fighter Squadron 154 (VFA-154), also known as the "Black Knights", is a United States Navy strike fighter squadron stationed at Naval Air Station
Lemoore.

 The Black Knights are an operational fleet squadron flying the F/A-18F Super Hornet.

 They are currently attached to Carrier Air Wing Eleven and deployed aboard the aircraft carrier USS "Nimitz" .

 Their tailcode is NH and their callsign is "Knight".
</item 1>

<item 2>
VAQ-135
Electronic Attack Squadron 135 (VAQ-135), known as the "Black Ravens", is a United States Navy electronic attack squadron that currently operates the EA-18G
Growler carrier-based electronic warfare jet aircraft.

 The squadron is permanently stationed at Naval Air Station Whidbey Island with a radio callsign of ""Thunder"".
</item 2>

<item 3>
VFA-2
Strike Fighter Squadron 2 (VFA-2) also known as the "Bounty Hunters" is a United States Navy F/A-18F Super Hornet strike fighter squadron based at Naval Air
Station Lemoore, California.

 Their tail code is NE and their callsign is "Bullet".

 They are attached to Carrier Air Wing 2 (CVW-2), a composite unit made up of a wide array of aircraft performing a variety of combat and support missions that
deploy aboard the "Carl Vinson" .
</item 3>

<item 4>
Boeing EA-18G Growler
The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.

 The EA-18G replaced the Northrop Grumman EA-6B Prowlers in service with the United States Navy.

 The Growler's electronic warfare capability is primarily provided by Northrop Grumman.

 The EA-18G began production in 2007 and entered operational service with the US Navy in late 2009.

 Australia has also purchased twelve EA-18Gs, which entered service with the Royal Australian Air Force in 2017.
</item 4>

<item 5>
Boeing F/A-18E/F Super Hornet
The Boeing F/A-18E and F/A-18F Super Hornet are twin-engine carrier-capable multirole fighter aircraft variants based on the McDonnell Douglas F/A-18 Hornet.

 The F/A-18E single-seat and F/A-18F tandem-seat variants are larger and more advanced derivatives of the F/A-18C and D Hornet.

 The Super Hornet has an internal 20 mm M61 rotary cannon and can carry air-to-air missiles and air-to-surface weapons.

 Additional fuel can be carried in up to five external fuel tanks and the aircraft can be configured as an airborne tanker by adding an external air refueling
system.
</item 5>

<item 6>
VAQ-138
Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at
Naval Air Station Whidbey Island, Washington.

 Their radio callsign is "RAMPAGE".
</item 6>

<item 7>
The EA-18G Growler, based on the two-seat F/A-18F Super Hornet, replaced the US Navy’s EA-6B and entered service in 2009; Australia operates 12. US Navy Growler
squadrons include VAQ-138, -209 and -135. Australia’s Amberley is its largest operational airbase, hosting Super Hornet squadrons and other units, including No.
1 Squadron, part of No. 82 Wing. The material
</item 7>

<item 8>
No. 82 Wing RAAF
No. 82 Wing is the strike and reconnaissance wing of the Royal Australian Air Force (RAAF).

 It is headquartered at RAAF Base Amberley, Queensland.

 Coming under the control of Air Combat Group, the wing operates F/A-18F Super Hornet multirole fighters and Pilatus PC-9 forward air control aircraft.

 Its units include Nos. 1 and 6 Squadrons, operating the Super Hornet, and No. 4 Squadron, operating the PC-9.
</item 8>

<item 9>
No. 1 Squadron RAAF
No. 1 Squadron is a Royal Australian Air Force (RAAF) squadron headquartered at RAAF Base Amberley, Queensland.

 Controlled by No. 82 Wing, it is equipped with Boeing F/A-18F Super Hornet multi-role fighters.

 The squadron was formed under the Australian Flying Corps in 1916 and saw action in the Sinai and Palestine Campaigns during World War I.

 It flew obsolete Royal Aircraft Factory B.E.2s, B.E.12s, Martinsyde G.100s and G.102s, as well as Airco DH.6s, Bristol Scouts and Nieuport 17s, before
re-equipping with the R.E.8 in October 1917 and finally the Bristol Fighter in December.

 Its commanding officer in 1917–18 was Major Richard Williams, later known as the "Father of the RAAF".

 Disbanded in 1919, No. 1 Squadron was re-formed on paper as part of the RAAF in 1922, and re-established as an operational unit three years later.
</item 9>

<item 10>
VAQ-209
Electronic Attack Squadron 209 (VAQ-209) is a United States Navy Reserve electronic attack squadron.

 Known as the "Star Warriors", the squadron flies the EA-18G Growler carrier-based electronic warfare jet aircraft.

 Based at Naval Air Station Whidbey Island, WA, it is assigned to the Tactical Support Wing.
</item 10>
=== END ITEM id=1d958d39819b278c ===

=== ITEM id=a24dd203acc9bdcb ===
Gold sentences and facts:
Sentence 0: Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
  Fact 0.0: Giuseppe Fortunino Francesco Verdi was born on 9 or 10 October 1813.
  Fact 0.1: Giuseppe Fortunino Francesco Verdi died on 27 January 1901.
  Fact 0.2: Giuseppe Fortunino Francesco Verdi was an Italian opera composer.
Sentence 1: Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
  Fact 1.0: Charles Louis Ambroise Thomas was born on 5 August 1811.
  Fact 1.1: Charles Louis Ambroise Thomas died on 12 February 1896.
  Fact 1.2: Charles Louis Ambroise Thomas was a French composer.
  Fact 1.3: Charles Louis Ambroise Thomas is best known for his opera 'Mignon' (1866).
  Fact 1.4: Charles Louis Ambroise Thomas is best known for his opera 'Hamlet' (1868).
  Fact 1.5: The opera 'Hamlet' (1868) is after Shakespeare.
  Fact 1.6: Charles Louis Ambroise Thomas is best known as Director of the Conservatoire de Paris.
  Fact 1.7: Charles Louis Ambroise Thomas was Director of the Conservatoire de Paris from 1871 till his death.

Retrieved items:
<item 1>
Ferdinando Provesi
Ferdinando Angelo Maria Provesi (1770 – 1833) was a native of Parma, Italy.

 He was regarded as one of the greatest Italian opera composers of the era .

 Provesi is best known as being an early tutor of Giuseppe Verdi when he was the "Maestro di cappella" (master of music) at the St. Bartolomeo cathedral in
Busseto (the town very close to Le Roncole, the village where Verdi was born.)

 Provesi was also director of the municipal music school and local Philharmonic Society.

 He began teaching Verdi in 1824 when the future composer was 11 years old.
</item 1>

<item 2>
Giuseppe Verdi (film)
Giuseppe Verdi is a 1938 Italian biographical film directed by Carmine Gallone and starring Fosco Giachetti, Gaby Morlay and Germana Paolieri.

 The film portrays the life of the composer Giuseppe Verdi (1813-1901).

 The casting of Giachetti as Verdi was intended to emphasise the composer's patriotism, as he had recently played patriotic roles in films such as "The White
Squadron".

 The film was made at the Cinecittà Studios in Rome.

 The film is also known by the alternative title The Life of Giuseppe Verdi.
</item 2>

<item 3>
List of major opera composers
This list provides a guide to opera composers, as determined by their presence on a majority of compiled lists of significant opera composers.

 (See the "Lists Consulted" section for full details.)

 The composers run from Jacopo Peri, who wrote the first ever opera in late 16th century Italy, to John Adams, one of the leading figures in the contemporary
operatic world.

 The brief accompanying notes offer an explanation as to why each composer has been considered major.

 Also included is a section about major women opera composers, compiled from the same lists.

 For an introduction to operatic history, see opera.

 The organisation of the list is by birthdate.
</item 3>

<item 4>
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 4>

<item 5>
Alessandro Scarlatti
Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.

 He is considered the founder of the Neapolitan school of opera.

 He was the father of two other composers, Domenico Scarlatti and Pietro Filippo Scarlatti.
</item 5>

<item 6>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 6>

<item 7>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 7>

<item 8>
Teatro Giuseppe Verdi
Teatro Giuseppe Verdi (the Giuseppe Verdi Theatre) is a small opera house located in a wing of the Rocca dei Marchesi Pallavicino on the Piazza Giuseppe Verdi
in Busseto, Italy, a town closely associated with the life of the opera composer, Giuseppe Verdi.

 From the 13th century, the “rocca” or “fortress” was the family’s palace; it is now the city hall after being acquired by the municipality in 1856.

 The theatre opened on 15 August 1868 and seats 300.
</item 8>

<item 9>
Ambroise Thomas
Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after
Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
</item 9>

<item 10>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 10>
=== END ITEM id=a24dd203acc9bdcb ===

=== ITEM id=fba442969a331e42 ===
Gold sentences and facts:
Sentence 0: Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.
  Fact 0.0: Mulled wine is a beverage.
  Fact 0.1: Mulled wine is usually made with red wine.
  Fact 0.2: Mulled wine is usually made with various mulling spices.
  Fact 0.3: Mulled wine is sometimes made with raisins.
Sentence 1:  It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.
  Fact 1.0: Mulled wine is served hot or warm.
  Fact 1.1: Mulled wine is alcoholic.
  Fact 1.2: There are non-alcoholic versions of mulled wine.
Sentence 2: Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
  Fact 2.0: Blue Lagoon is a popular summer cocktail.
  Fact 2.1: Blue Lagoon features blue Curaçao.

Retrieved items:
<item 1>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 1>

<item 2>
Mulled wine
Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.

 It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.

 It is a traditional drink during winter, especially around Christmas.
</item 2>

<item 3>
Mulling spices
Mulling spices is a spice mixture used in drink recipes.

 The spices are usually used to spice hot apple cider, mulled wine, wassail, and other drinks (such as juices) during the autumn or winter.

 A "mulled" drink is a drink which has been prepared with these spices (usually through heating the drink in a pot with mulling spices and then straining).

 Mulling spices may also be added to the brewing process to make spiced beer.
</item 3>

<item 4>
Blue Lagoon (cocktail)
Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
</item 4>

<item 5>
Oz and Hugh Drink to Christmas
Oz and Hugh Drink to Christmas is a BBC television programme in which wine personality Oz Clarke and comedian Hugh Dennis travel through Britain to sample a
wide array of seasonal Christmas beverages, including whisky, winter ales, mulled wine, wassail, sloe gin, Buck's Fizz, Port wine and Sherry.

 Upon its 20 December 2009 broadcast on BBC Two, it had a viewership of approximately 2.4 million with an audience share of 9%.
</item 5>

<item 6>
Volcano bowl
Volcano bowls are ceramic drinkware originally associated with mid-20th century American tiki bars and tropical-themed restaurants.

 Drinks served in volcano bowls are typically rum-based, mixed with tropical fruit juices and other liquors such as brandy, vodka, and triple sec, and garnished
with fruit.

 The Flaming Volcano cocktail is especially associated with this drinkware.
</item 6>

<item 7>
Feuerzangenbowle
Feuerzangenbowle (] ) is a traditional German alcoholic drink for which a rum-soaked sugarloaf is set on fire and drips into mulled wine.

 It is often part of a Christmas or New Year's Eve tradition.

 The name translates literally to "fire-tongs punch".
</item 7>

<item 8>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 8>

<item 9>
The Blue Lagoon (novel)
The Blue Lagoon is a romance novel written by Henry De Vere Stacpoole and was first published by T. Fisher Unwin in 1908.

 It is the first novel of the "Blue Lagoon" trilogy, which also includes "The Garden of God" (1923) and "The Gates of Morning" (1925).

 The novel has inspired several film adaptations, most notably "The Blue Lagoon" starring Brooke Shields as Emmeline and Christopher Atkins as Richard ("Dicky"
in the book).
</item 9>

<item 10>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 10>
=== END ITEM id=fba442969a331e42 ===

=== ITEM id=a1f00a9fbc9d7d96 ===
Gold sentences and facts:
Sentence 0:  Along with competing in the league, the club will also participate in the FA Cup, EFL Cup and EFL Trophy.
  Fact 0.0: The club will compete in the league.
  Fact 0.1: The club will also participate in the FA Cup.
  Fact 0.2: The club will also participate in the EFL Cup.
  Fact 0.3: The club will also participate in the EFL Trophy.
Sentence 1: The EFL Cup (referred to historically, and colloquially, as simply the League Cup), currently known as the Carabao Cup for sponsorship reasons, is an annual knockout football competition in men's domestic English football.
  Fact 1.0: The EFL Cup is historically referred to as the League Cup.
  Fact 1.1: The EFL Cup is colloquially referred to as simply the League Cup.
  Fact 1.2: The EFL Cup is currently known as the Carabao Cup for sponsorship reasons.
  Fact 1.3: The EFL Cup is an annual knockout football competition.
  Fact 1.4: The EFL Cup is a competition in men's domestic English football.

Retrieved items:
<item 1>
2007–08 Wigan Athletic F.C. season
The 2007–08 Wigan Athletic F.C. season was the club's 30th season in the Football League and their third season in the Premier League.
</item 1>

<item 2>
1978–79 Wigan Athletic F.C. season
The 1978–79 season was the 41st season in the history of Wigan Athletic F.C. and their first as a professional club in the Football League.

 After finishing 2nd in the Northern Premier League during the previous season, the club was nominated to apply for Football League status, and were elected
into the league to replace Southport.

 After a poor start, the club exceeded expectations in the league, and towards the end of the season, a second consecutive promotion seemed possible.

 The club eventually fell short, finishing the season in 6th place with a total of 55 points.
</item 2>

<item 3>
2017–18 Wigan Athletic F.C. season
The 2017–18 season is Wigan Athletic's 86th year in existence and their first back in League One, after being relegated the previous season.

 Along with competing in the league, the club will also participate in the FA Cup, EFL Cup and EFL Trophy.

 Paul Cook was appointed as manager on 31 May 2017, signing a 3 year deal, following the departure of much of the previous coaching staff the day before.
</item 3>

<item 4>
EFL Cup
The EFL Cup (referred to historically, and colloquially, as simply the League Cup), currently known as the Carabao Cup for sponsorship reasons, is an annual
knockout football competition in men's domestic English football.

 Organised by the English Football League (EFL), it is open to any club within the top four levels of the English football league system – 92 clubs in total –
comprising the top level Premier League, and the three divisions of the English Football League's own league competition (Championship, League One and League
Two).
</item 4>

<item 5>
List of Wigan Borough F.C. players
Wigan Borough F.C. was an English football club based in the town of Wigan.

 The club was founded in 1920 and joined the Lancashire Combination.

 In 1921, Borough turned professional when their application was accepted to play in the inaugural season of the newly formed Football League Third Division
North.

 The team played in the Football League for ten seasons, with their most successful season coming in 1928–29, finishing fourth in the league and reaching the
third round of the FA Cup.

 Wigan Borough folded during the 1931–32 season due to financial problems, and League football did not return to the town until Wigan Athletic F.C. were elected
into the Football League in 1978.
</item 5>

<item 6>
2013–14 Midlands football leagues (levels 9–10)
The 2013–14 Midland Football Alliance season was the 20th and final in the history of Midland Football Alliance, a football competition in England.

 The 2013–14 Midland Football Combination season (known as the 2013–14 Athium Midland Football Combination for sponsorship reasons) was the 77th and final in
the history of Midland Football Combination and it was a feeder to the Midland Football Alliance.

 The 2013–14 West Midlands (Regional) League season was the 114th in the history of the West Midlands (Regional) League.

 It was also a feeder league to the Midland Football Alliance.

 At the end of the season the Midland Alliance and the Midland Combination merged to form the Midland Football League.
</item 6>

<item 7>
Springfield Park (Wigan)
Springfield Park was a multi-purpose stadium in Wigan, Greater Manchester.

 It was the home ground of Wigan Athletic F.C. until the club moved to the new JJB Stadium (now DW Stadium) after the 1998–99 season.

 At its largest, the stadium held 40,000.

 In its 102-year existence the ground only saw 32 years as a Football League venue, 11 years for Wigan Borough F.C. and 21 years for Wigan Athletic FC, before
it was demolished to make way for a housing estate in 1999.
</item 7>

<item 8>
2006–07 Wigan Athletic F.C. season
The 2006–07 Wigan Athletic F.C. season was the club's 29th season in the Football League and their second season in the Premier League.
</item 8>

<item 9>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 9>

<item 10>
1979–80 Wigan Athletic F.C. season
The 1979–80 season was the 42nd season in the history of Wigan Athletic F.C. and their second as a professional club in the Football League.
</item 10>
=== END ITEM id=a1f00a9fbc9d7d96 ===

=== ITEM id=a315e85a8f7ae726 ===
Gold sentences and facts:
Sentence 0:  He creates bespoke millinery under the J Smith Esquire brand for a portfolio of private clients that includes Angelina Jolie, Brad Pitt and Amal Clooney.
  Fact 0.0: This person creates bespoke millinery under the J Smith Esquire brand.
  Fact 0.1: This person creates bespoke millinery for a portfolio of private clients.
  Fact 0.2: The portfolio of private clients includes Angelina Jolie.
  Fact 0.3: The portfolio of private clients includes Brad Pitt.
  Fact 0.4: The portfolio of private clients includes Amal Clooney.
Sentence 1: Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and human rights.
  Fact 1.0: Amal Clooney is also known by the name Alamuddin (née Alamuddin).
  Fact 1.1: Amal Clooney's Arabic name is أمل علم الدين.
  Fact 1.2: Amal Clooney was born on 3 February 1978.
  Fact 1.3: Amal Clooney is a barrister at Doughty Street Chambers.
  Fact 1.4: Amal Clooney specialises in international law.
  Fact 1.5: Amal Clooney specialises in human rights.

Retrieved items:
<item 1>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 1>

<item 2>
Pupil master
A pupil master ('pupilmaster' or 'pupil-master') or in the case of a female barrister, 'pupil mistress' etc., is the former name given to an experienced
barrister who a pupil shadows during their pupillage.

 The term pupil master or mistress has now been replaced by the term 'pupil supervisor'.

 Barristers are called to the Bar via one of the four Inns of Court upon successful completion of the BPTC and having undertaken a required number of
"qualifying sessions" in their chosen Inn of Court.

 In most cases, the newly called barrister is then required to undertake training for a period of at least a year before the barrister can start their own
private practice.

 This training period is known as pupillage, usually split into two periods of six months known as "sixes".

 The first "six" is a non-practising six, during which the pupil will shadow their pupil master; the second is usually a practising "six", when the pupil, with
their pupil master's permission, can undertake the supply of legal services and exercise rights of audience in court.
</item 2>

<item 3>
Pupil master
Occasionally, a pupil barrister may undertake a third "six", extending the training period a further six months.

 At the end of pupillage, to continue practising the law, a barrister may attempt to become a tenant in a set of barristers' chambers, or find a position as an
employed barrister.
</item 3>

<item 4>
Barbados Bar Association
The Barbados Bar Association is a voluntary association of attorneys in Barbados who practise at the independent bar as barristers and Queen's Counsel.

 It was created by the Barbados Bar Association Act of 1940.
</item 4>

<item 5>
Shahram Taghavi
Shahram Taghavi is a barrister practising in the United Kingdom.

 He is a specialist in Judicial Review, Human Rights, Immigration and EU law.

 He was called to the England & Wales bar in 1994.

 He is a partner at Article 1, a law firm specialising in immigration, EU freedoms of movement and human rights law.

 He practised as an independent barrister at Doughty Street Chambers specialising in immigration and public law.

 He was head of both the Immigration and Public Law departments at Simons Muirhead & Burton solicitors.

 He was a Senior Barrister at Bates Wells & Braithwaite LLP as a member of their Public & Regulatory and Immigration departments, Deputy Head of Immigration at
Lewis Silkin LLP, Head of Immigration at Charles Russell LLP (now Charles Russell Speechly LLP) and National Head of Immigration at Grant Thornton LLP.
</item 5>

<item 6>
Mónica Feria Tinta
Mónica Feria Tinta is a leading public international lawyer.

 She practises as a Barrister at 20 Essex Street Chambers.

 In 2000 she became the first Peruvian-born lawyer to receive the Diploma of the Hague Academy of International Law in history.

 Her litigation work led to the first international human rights court decision ordering the prosecution of a former Head of State for crimes under
international law.

 She was awarded the Inge Genefke International Award for her work as an international lawyer in 2006 and was co-recipient of the Gruber Justice Prize 2007, for
her contributions advancing the cause of justice as delivered through the legal system.

 She is a member of the Bar of England and Wales, and the American Society of International Law.
</item 6>

<item 7>
The entries cover British and Irish figures, families, and titles: Edmund Bartley-Denniss, barrister, Conservative MP, Freemason, and cycling pioneer; several
Mortimers and their royal claims, marriages, and fortunes; the 1405 Tripartite Indenture dividing England and Wales; and Lady Elizabeth Percy. They also trace
Richard Sherman and Miles Barne’s parliamentary careers, peerages including Plunket, Norbury, Cardross, Cowper, and Galway, North
</item 7>

<item 8>
City Cruises runs year-round, wheelchair-accessible Thames sightseeing services between major London piers, plus private charters, the London Showboat dinner
cruise and permanently moored Hispaniola restaurant. The remaining entries describe London locations and institutions: Doughty Street Chambers’ legal work and
Doughty Street’s history; Westminster’s landmarks and former parishes; Abbey Road and its Beatles association; Westminster Underground station; Temple Bar’s
ceremonial role and Wren gateway; and the former Westminster Magistrates’
</item 8>

<item 9>
Doughty Street Chambers
Doughty Street Chambers is a British set of barristers' chambers situated in Bristol, Manchester and London's Doughty Street, undertaking criminal justice,
public law, immigration, employment, human rights and civil liberties work.
</item 9>

<item 10>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 10>
=== END ITEM id=a315e85a8f7ae726 ===

=== ITEM id=100db07a3e7e8e49 ===
Gold sentences and facts:
Sentence 0: Montague (Monty) Banks (15 July 1897 [registered on 18 July 1897] – 7 January 1950 born Mario Bianchi) was an Italian comedian and film director who achieved success in the United States and in England.
  Fact 0.0: Montague (Monty) Banks was born Mario Bianchi.
  Fact 0.1: Montague (Monty) Banks was born 15 July 1897.
  Fact 0.2: Montague (Monty) Banks' birth was registered on 18 July 1897.
  Fact 0.3: Montague (Monty) Banks died on 7 January 1950.
  Fact 0.4: Montague (Monty) Banks was an Italian comedian.
  Fact 0.5: Montague (Monty) Banks was an Italian film director.
  Fact 0.6: Montague (Monty) Banks achieved success in the United States.
  Fact 0.7: Montague (Monty) Banks achieved success in England.
Sentence 1: George Seaton (April 17, 1911 – July 28, 1979) was an American screenwriter, playwright, film director and producer, and theatre director.
  Fact 1.0: George Seaton was born April 17, 1911.
  Fact 1.1: George Seaton died July 28, 1979.
  Fact 1.2: George Seaton was an American screenwriter.
  Fact 1.3: George Seaton was an American playwright.
  Fact 1.4: George Seaton was an American film director.
  Fact 1.5: George Seaton was an American film producer.
  Fact 1.6: George Seaton was an American theatre director.

Retrieved items:
<item 1>
George Seaton
George Seaton (April 17, 1911 – July 28, 1979) was an American screenwriter, playwright, film director and producer, and theatre director.
</item 1>

<item 2>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 2>

<item 3>
Keep Your Seats, Please
Keep Your Seats, Please is a 1936 British comedy film directed by Monty Banks and starring George Formby, Florence Desmond and Alastair Sim.

 It marked the film debut of the child star Binkie Stuart.

 The film was made by Associated Talking Pictures.
</item 3>

<item 4>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 4>

<item 5>
The Church Mouse
The Church Mouse is a 1934 British comedy film directed by Monty Banks and starring Laura La Plante, Ian Hunter and Edward Chapman.

 It was made by the British subsidiary of Warner Brothers at the company's Teddington Studios.

 It was made as a more expensive production than much of the studio's low-budget quota quickie output.
</item 5>

<item 6>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
</item 6>

<item 7>
Joey Bishop
Joseph Abraham Gottlieb (February 3, 1918 – October 17, 2007), known professionally as Joey Bishop, was an American entertainer who appeared on television as
early as 1948 and eventually starred in his own weekly comedy series playing a talk show host, then later hosted a late night talk show.

 He later became a member of the "Rat Pack" with Frank Sinatra, Peter Lawford, Sammy Davis, Jr., and Dean Martin.
</item 7>

<item 8>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 8>

<item 9>
Mark Benton
Mark Benton (born 16 November 1965) is an English actor, known for his roles as Eddie in "Early Doors", Howard in "Northern Lights" and Martin Pond in
"Barbara".

 Benton has also starred in BBC One drama, "Waterloo Road" as maths teacher, Daniel Chalk from 2011 to 2013 and again in 2014.
</item 9>

<item 10>
Monty Banks
Montague (Monty) Banks (15 July 1897 [registered on 18 July 1897] – 7 January 1950 born Mario Bianchi) was an Italian comedian and film director who achieved
success in the United States and in England.
</item 10>
=== END ITEM id=100db07a3e7e8e49 ===
