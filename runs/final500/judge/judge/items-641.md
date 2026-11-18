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

This file has 13 items.

=== ITEM id=e13fefee6e0d1f2a ===
Gold sentences and facts:
Sentence 0: Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
  Fact 0.0: Leycesteria is a genus of flowering plants.
  Fact 0.1: Leycesteria is in the honeysuckle family Caprifoliaceae.
  Fact 0.2: Leycesteria is native to temperate Asia in the Himalaya.
  Fact 0.3: Leycesteria is native to southwestern China.
Sentence 1: Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.
  Fact 1.0: Anigozanthos is a small genus of Australian plants.
  Fact 1.1: Anigozanthos is in the Bloodwort family Haemodoraceae.

Retrieved items:
<item 1>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 1>

<item 2>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 2>

<item 3>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 3>

<item 4>
Xanthosoma
Xanthosoma is a genus of flowering plants in the arum family, Araceae.

 The genus is native to tropical America but widely cultivated and naturalized in other tropical regions.

 Several are grown for their starchy corms, an important food staple of tropical regions, known variously as "malanga", "otoy", "otoe", cocoyam (or new
cocoyam), "tannia", "tannier", "yautía", "macabo", "ocumo", "macal", "taioba", "dasheen", "quequisque", "ʻ ape" and (in Papua New Guinea) as Singapore taro
("taro kongkong").

 Many other species (including especially "X. roseum") are used as ornamental plants, and in popular horticultural literature are known as ‘ape or elephant ear
(from the purported resemblance of the leaf to an elephant's ear), although the latter name is sometimes also applied to members with similar appearance and
uses in the closely related genera "Caladium", "Colocasia" (i.e., taro), and "Alocasia".
</item 4>

<item 5>
Anigozanthos preissii
Anigozanthos preissii, the Albany cat's paw, is a herbaceous plant species in the family Haemodoraceae, endemic to Western Australia.
</item 5>

<item 6>
Anigozanthos
Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.

 The 11 species and several subspecies are commonly known as kangaroo paw and catspaw depending on the shape of their flowers.

 A further species, previously identified as "Anigozanthos fuliginosus" and commonly known as the black kangaroo paw, has been transferred to its own monotypic
genus and is now known as "Macropidia fuliginosa".
</item 6>

<item 7>
Leycesteria formosa
Leycesteria formosa (Himalayan honeysuckle, flowering nutmeg, Himalaya nutmeg or pheasant berry) is a deciduous shrub in the family Caprifoliaceae, native to
the Himalaya and southwestern China.

 It is considered a noxious invasive species in Australia, New Zealand, the neighboring islands of Micronesia, and some other places.
</item 7>

<item 8>
Lepidozamia comprises two Australian rainforest cycads (2n=18). Orange Field, a Sri Lankan tea factory founded in 1950, makes about 500,000 kg of black tea
annually for Middle Eastern and CIS markets, sourcing from 750 smallholders. Lithocarpus has 334 accepted East/Southeast Asian species; three Java endemics
listed here are endangered, while L. glaber occurs in Japan, China and Taiwan. Oaks (Querc
</item 8>

<item 9>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 9>

<item 10>
Anigozanthos humilis
Anigozanthos humilis is a species of "Anigozanthos" in the family Haemodoraceae, known as common cat's paw or catspaw.

 This flowering perennial plant is endemic to Southwest Australia and widespread in its open forests.
</item 10>
=== END ITEM id=e13fefee6e0d1f2a ===

=== ITEM id=f55bf3ab7b87c5f2 ===
Gold sentences and facts:
Sentence 0: Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.
  Fact 0.0: Orlando Brown was born on December 4, 1987.
  Fact 0.1: Orlando Brown is an American actor.
  Fact 0.2: Orlando Brown is a voice actor.
  Fact 0.3: Orlando Brown is a rapper.
  Fact 0.4: Orlando Brown is a singer.
Sentence 1:  He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
  Fact 1.0: Orlando Brown is best known for his role as Eddie Thomas in "That's So Raven".
  Fact 1.1: Orlando Brown is best known for his role as 3J Winslow in "Family Matters".
  Fact 1.2: Orlando Brown is best known for his role as Tiger in "Major Payne".
  Fact 1.3: Orlando Brown is best known for his role as Max in "Two of a Kind".
  Fact 1.4: Orlando Brown is best known for his role as Damey Wayne in the short-lived Waynehead.
  Fact 1.5: Orlando Brown is best known for his role as Dobbs in "Max Keeble's Big Move".
  Fact 1.6: Orlando Brown is best known for his role as Frankie in "Eddie's Million Dollar Cook Off".
Sentence 2: Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17, 1998.
  Fact 2.0: Family Matters is an American sitcom.
  Fact 2.1: Family Matters originated on ABC.
  Fact 2.2: Family Matters originated on ABC from September 22, 1989 to May 9, 1997.
  Fact 2.3: Family Matters moved to CBS after its ABC run.
  Fact 2.4: Family Matters was on CBS from September 19, 1997 to July 17, 1998.
Sentence 3:  A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.
  Fact 3.0: Family Matters is a spin-off of "Perfect Strangers."
  Fact 3.1: Family Matters revolves around the Winslow family.
  Fact 3.2: The Winslow family is a middle-class African American family.
  Fact 3.3: The Winslow family lives in Chicago, Illinois.

Retrieved items:
<item 1>
Lamman Rucker
Lamman Rucker (born October 6, 1971) is an American actor.

 Rucker began his career on the daytime soap operas "As the World Turns" and "All My Children", before roles in Tyler Perry's films "Why Did I Get Married?

", "Why Did I Get Married Too?

", and "Meet the Browns", and its television adaptation.

 In 2016, he began starring as Jacob Greenleaf in the Oprah Winfrey Network drama series, "Greenleaf".
</item 1>

<item 2>
Family Matters
Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17,
1998.

 A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.

 Midway through the first season, the show introduced the Winslows' nerdy neighbor Steve Urkel (Jaleel White), who quickly became its breakout character and
eventually the show's main character.

 Having run for nine seasons, "Family Matters" became the second longest-running non-animated US sitcom with a predominantly African American cast, behind only
"The Jeffersons" (11).

 Having aired 215 episodes, "Family Matters" is ranked third, behind only "Tyler Perry's House of Payne" (254), and "The Jeffersons" (253).
</item 2>

<item 3>
Robert Oliveri
Robert Dane Oliveri (born April 28, 1978) is a former American actor.

 He is best known for his portrayal of Nick Szalinski in the 1989 Disney film, "Honey, I Shrunk the Kids", for which he was nominated for a Young Artist Award
and a Saturn Award.

 He reprised the role in the 1992 sequel, "Honey, I Blew Up the Kid" and the 3D film, "Honey, I Shrunk the Audience".

 He is also known as Kevin, Winona Ryder's little brother, in Tim Burton's "Edward Scissorhands".
</item 3>

<item 4>
Eric Brown (actor)
Eric Brown (born December 17, 1964) is a former American film and television actor.

 Brown is best known for his role as Phillip Fillmore in the 1981 coming of age film "Private Lessons" and as Vinton "Buzz" Harper, Jr. in the NBC version of
the American sitcom "Mama's Family".
</item 4>

<item 5>
Darius McCrary
Darius Creston McCrary (born May 1, 1976) is an American film and television actor and singer.

 He is best known for his role as Edward "Eddie" Winslow, the oldest child of Carl and Harriette Winslow on the ABC/CBS television sitcom "Family Matters "which
ran from 1989–1998.

 Another one of McCrary's notable roles was as Scam in the 1987 comedy film "Big Shots", which was his film debut.

 He provided the voice of Jazz in the 2007 "Transformers" reboot.

 From December 2009 to October 2011, he portrayed photographer Malcolm Winters on the CBS daytime drama "The Young and the Restless".
</item 5>

<item 6>
Michael Jai White
Michael Jai White (born November 10, 1967) is an American actor and martial artist who has appeared in numerous films and television series.

 He is the first African American to portray a major comic book superhero in a major motion picture, having starred as Al Simmons, the protagonist in the 1997
film "Spawn".

 White appeared as Marcus Williams in the Tyler Perry films "Why Did I Get Married?

" and "Why Did I Get Married Too?

", and starred as the character on the TBS/OWN comedy-drama television series "Tyler Perry's For Better or Worse".

 White portrayed Jax Briggs in "".

 White also portrayed boxer Mike Tyson in the 1995 HBO television movie "Tyson", and starred as Black Dynamite.
</item 6>

<item 7>
Jaimee Foxworth
Jaimee Monae Foxworth (born December 17, 1979) is an American actress and model.

 She is best known for her role of Judy Winslow, the youngest daughter of Carl and Harriette Winslow on the ABC/CBS sitcom "Family Matters".

 She later transitioned to pornographic films using the name Crave, starring in several titles.
</item 7>

<item 8>
Orlando Brown (actor)
Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.

 He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne
in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
</item 8>

<item 9>
Randy J. Goodwin
Randy J. Goodwin, born as Ivan Jerome Goodwin, is a Hollywood actor.

 He is perhaps best known for his role as "Kennedy Winslow" on "Fast Track" and "Max Ellis" on the show "Abby", as well as his recurring roles on "Girlfriends"
as "Davis Hamilton" and "The Vampire Diaries" as "Dr. Jonas Martin".
</item 9>

<item 10>
Tyler Perry's For Better or Worse
Tyler Perry's For Better or Worse is an American television sitcom created, written and executive produced by Tyler Perry.

 The series is based on and is a TV spin-off to Perry's 2007 film "Why Did I Get Married?

" and its 2010 sequel "Why Did I Get Married Too?

".

 The show premiered on TBS on November 25, 2011 and ended on July 22, 2017.

 Led by the comical, over-the-top antics of Marcus and Angela Williams, the ensemble follows three couples: Marcus and Angela, Joseph and Leslie, and Richard
and Keisha who are at various stages of their relationships as they go through the ups-and-downs of married life and dating.

 The third season of the series premiered on September 18, 2013, on the Oprah Winfrey Network.
</item 10>
=== END ITEM id=f55bf3ab7b87c5f2 ===

=== ITEM id=e2c4e5beef070b3b ===
Gold sentences and facts:
Sentence 0:  It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.
  Fact 0.0: The work is the third of Gluck's so-called reform operas for Vienna.
  Fact 0.1: The work follows 'Alceste' and 'Orfeo ed Euridice' among Gluck's reform operas for Vienna.
  Fact 0.2: The work is the least often performed of the three reform operas for Vienna.
Sentence 1: Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of Orpheus, set to a libretto by Ranieri de' Calzabigi.
  Fact 1.0: Orfeo ed Euridice is an opera composed by Christoph Willibald Gluck.
  Fact 1.1: Orfeo ed Euridice is based on the myth of Orpheus.
  Fact 1.2: Orfeo ed Euridice is set to a libretto by Ranieri de' Calzabigi.
  Fact 1.3: The French version of Orfeo ed Euridice is titled Orphée et Eurydice.
  Fact 1.4: The English title of Orfeo ed Euridice is 'Orpheus and Eurydice'.
Sentence 2:  The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.
  Fact 2.0: The piece was first performed at the Burgtheater in Vienna on 5 October 1762.
  Fact 2.1: The first performance of the piece was in the presence of Empress Maria Theresa.

Retrieved items:
<item 1>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 1>

<item 2>
Das geheime Königreich
Das geheime Königreich ("The Secret Kingdom") is an opera in one act with words and music by Ernst Krenek, his Op.

 50 and the second of three one-acters (with "Der Diktator" and "Schwergewicht, oder Die Ehre der Nation") which premiered on 6 May 1928 at the Hessisches
Staatstheater Wiesbaden as part of the Mai-Festwoche Wiesbaden.

 It is subtitled "Märchenoper in 1 Akt " (fairy-tale opera in one act) and has been called a satirical fairy-tale opera.
</item 2>

<item 3>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 3>

<item 4>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 4>

<item 5>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 5>

<item 6>
Maria Theresa (film)
Maria Theresa (German: Maria Theresia) is a 1951 Austrian historical drama film directed by  Emil E. Reinert and starring  Paula Wessely, Fred Liewehr and
Marianne Schönauer.

 It portrays the life of the eighteenth century Habsburg Empress Maria Theresa.
</item 6>

<item 7>
The entries span opera, music, theatre, literature and Renaissance history. Bernhard Bötel sang roles from Mozart to Verdi; operas range from Gluck and Hasse to
Rossini and later composers. Commedia dell’arte’s masked stock characters, improvisation and lazzi are described. Musical highlights include Tchaikovsky’s
Second Piano Concerto, Debussy’s *Masques* and Bartók’s quartets. Other topics include Verdi’s Busseto theatre,
</item 7>

<item 8>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 8>

<item 9>
Bernhard Bötel’s roles ranged from Mozart’s Belmonte to Verdi’s Duke of Mantua. The entries survey operas across eras: Gluck’s reform works *Orfeo ed Euridice*
(1762) and *Paride ed Elena* (1770); Hasse’s *Romolo ed Ersilia* (1765); Rossini’s *Mosè in Egitto* (1818); and later works by Donizetti, Bernstein,
</item 9>

<item 10>
Paride ed Elena
Paride ed Elena (] ; "Paris and Helen") is an opera by Christoph Willibald Gluck.

 It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.

 Like its predecessors, the libretto was written by Ranieri de' Calzabigi.

 The opera tells the story of the events between the Judgment of Paris and the flight of Paris and Helen to Troy.

 It was premiered at the Burgtheater in Vienna on 3 November 1770.
</item 10>
=== END ITEM id=e2c4e5beef070b3b ===

=== ITEM id=a92a3a922c629e41 ===
Gold sentences and facts:
Sentence 0: War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor at Kean University, and Albie Hecht.
  Fact 0.0: War/Dance is a 2007 American documentary film.
  Fact 0.1: War/Dance was written by Sean Fine and Andrea Nix Fine.
  Fact 0.2: War/Dance was directed by Sean Fine and Andrea Nix Fine.
  Fact 0.3: War/Dance was produced by Shine Global's Susan MacLaury.
  Fact 0.4: Susan MacLaury is a professor at Kean University.
  Fact 0.5: War/Dance was produced by Albie Hecht.
Sentence 1: The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy government and Nazi Germany during World War II.
  Fact 1.0: The Sorrow and the Pity is a documentary film.
  Fact 1.1: The Sorrow and the Pity's French title is Le Chagrin et la Pitié.
  Fact 1.2: The Sorrow and the Pity is in two parts.
  Fact 1.3: The Sorrow and the Pity was released in 1969.
  Fact 1.4: The Sorrow and the Pity was made by Marcel Ophüls.
  Fact 1.5: The Sorrow and the Pity is about the collaboration between the Vichy government and Nazi Germany.
  Fact 1.6: That collaboration occurred during World War II.

Retrieved items:
<item 1>
Unchained Memories
Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and
preserved in the WPA Slave Narrative Collection.

 This HBO film interpretation directed by Ed Bell and Thomas Lennon is a compilation of slave narratives, narrated by actors, emulating the original
conversation with the interviewer.

 The slave narratives may be the most accurate in terms of the everyday activities of the enslaved, serving as personal memoirs of more than two thousand former
slaves.

 The documentary depicts the emotions of the slaves and what they endured.

 The "Master" had the opportunity to sell, trade, or kill the enslaved, for retribution should one slave not obey.
</item 1>

<item 2>
Graph query row:
work: The Sorrow and the Pity
release_year: 1969
relationship: CREATED_BY
connected_fact: Marcel Ophüls
</item 2>

<item 3>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 3>

<item 4>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 4>

<item 5>
Graph query row:
work: War/Dance
release_year: 2007
relationship: CREATED_BY
connected_fact: Sean Fine
</item 5>

<item 6>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 6>

<item 7>
Agony (film)
Agony (Russian: Агония , "Agoniya " ; U.S. theatrical/DVD title "Rasputin") is a film by Elem Klimov, made c.1973-75 and released in Western and Central Europe
in 1982 (USA and Soviet Union 1985), after protracted resistance from Soviet authorities.

 The film is notable for its rich, sometimes baroque style, its sumptuous recreation of episodes from the final year of Imperial Russia and the psychological
portraits of Grigori Rasputin and the Imperial family.
</item 7>

<item 8>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 8>

<item 9>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 9>

<item 10>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 10>
=== END ITEM id=a92a3a922c629e41 ===

=== ITEM id=b808351ab94c3b73 ===
Gold sentences and facts:
Sentence 0: The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical creature of the forest, or Rå, in Swedish folklore.
  Fact 0.0: The skogsrå is also called Forest Rå.
  Fact 0.1: The skogsrå is also called Skogsfrun (The Mistress of the Forest).
  Fact 0.2: The skogsrå is also called Skogssnuvan.
  Fact 0.3: The skogsrå is also called Skogsnymfen (Forest Nymph).
  Fact 0.4: The skogsrå is also called Råndan (the Rå).
  Fact 0.5: The skogsrå is also called huldra.
  Fact 0.6: The skogsrå was a mythical creature of the forest, also called Rå, in Swedish folklore.
Sentence 1: A hulder is a seductive forest creature found in Scandinavian folklore.
  Fact 1.0: A hulder is a seductive forest creature.
  Fact 1.1: Hulder creatures are found in Scandinavian folklore.

Retrieved items:
<item 1>
Gertrud Dahlgren
Gertrud Dahlgren (1931–2009) was a Swedish botanist.

 She was born at Klippan in Scania, and after graduating from university at Helsingborg went to the University of Lund for graduate studies.

 There she obtained her M.Sc.

 in chemistry and biology.

 She was married to fellow botanist Rolf Dahlgren (1932–1987), who was killed in a car accident.

 The couple had three children, Susanna, Helena and Fredrik.

 She died in December 2009 at the age of 78.
</item 1>

<item 2>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 2>

<item 3>
Jenny Nyström
Jenny Eugenia Nyström (13 or 15 June 1854 in Kalmar, Sweden – 17 January 1946 in Stockholm) was a painter and illustrator who is mainly known as the person who
created the Swedish image of the "jultomte" on numerous Christmas cards and magazine covers, thus linking the Swedish version of Santa Claus to the gnomes of
Scandinavian folklore.
</item 3>

<item 4>
Vörðr
In Norse mythology, a vǫrðr (pl.

 varðir or verðir — "warden," "watcher" or "caretaker") is a warden spirit, believed to follow from birth to death the soul ("hugr") of every person.

 In Old Swedish, the corresponding word is "varþer"; in modern Swedish "vård", and the belief in them remained strong in Scandinavian folklore up until the last
centuries.

 The English word '"wraith" is derived from "vǫrðr", while "" and "warden" are cognates.
</item 4>

<item 5>
Almaz (mythical beast)
The almaz (Chechen: "almaz" or "hun-sag"; Adyghe: "almesti"), roughly translated as various "feral forest-man" or "stone spirit", is a mythical beast that is
considered to be an evil forest creature with magical powers residing in its hair that exists in Chechen and Ingush folk beliefs and Circassian folk beliefs.

 The first "attestation" of it in writing was by a Bavarian captive of the Mongols, but it is present in the national folklore of Chechens and Ingush and of
Circassians.

 The male almaz is said to be hairy and hideous, and have an axe embedded in its chest, while the female is very beautiful with large breasts and golden hair,
and has a "favorite pastime" of dancing naked at night under the moon.

 The almaz is said to have magical powers residing in its hair, but if the hair is removed or even grabbed, it may be rendered helpless.
</item 5>

<item 6>
Hulder
A hulder is a seductive forest creature found in Scandinavian folklore.

 (Her name derives from a root meaning "covered" or "secret".)

 In Norwegian folklore, she is known as huldra ("the [archetypal] hulder", though folklore presupposes that there is an entire Hulder race and not just a single
individual).

 She is known as the skogsrå "forest spirit" or Tallemaja "pine tree Mary" in Swedish folklore, and ulda in Sámi folklore.

 Her name suggests that she is originally the same being as the völva divine figure Huld and the German Holda.
</item 6>

<item 7>
Spettekaka
Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also
in Halland.

 It is an important part of the Scanian culinary heritage.

 The name means "cake on a spit", and this describes the method of preparation: it is the Swedish variation on the spit cake.
</item 7>

<item 8>
Selkie (disambiguation)
A selkie is a mythological creature found in Faroese, Icelandic, Irish, and Scottish folklore.
</item 8>

<item 9>
Erotikon (1920 film)
Erotikon is a 1920 Swedish romantic comedy film directed by Mauritz Stiller, starring Tora Teje, Karin Molander, Anders de Wahl and Lars Hanson.

 It is based on the 1917 play "A kék róka" by Ferenc Herczeg.

 The story revolves around an entomology professor obsessed with the sexual life of bugs, and his easygoing wife who is courted by two suitors.
</item 9>

<item 10>
Skogsrå
The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical
creature of the forest, or Rå, in Swedish folklore.
</item 10>
=== END ITEM id=b808351ab94c3b73 ===

=== ITEM id=a33a0e6a7520168e ===
Gold sentences and facts:
Sentence 0:  It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).
  Fact 0.0: The company began in 1995.
  Fact 0.1: The company began as the merger of two take-and-bake pizza companies.
  Fact 0.2: One of the merged companies was Papa Aldo's Pizza.
  Fact 0.3: Papa Aldo's Pizza was founded in 1981.
  Fact 0.4: The other merged company was Murphy's Pizza.
  Fact 0.5: Murphy's Pizza was founded in 1981.
Sentence 1: Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.
  Fact 1.0: Gino's Pizza and Spaghetti is a restaurant chain.
  Fact 1.1: Gino's Pizza and Spaghetti has 40 locations.
  Fact 1.2: Most of Gino's Pizza and Spaghetti's locations are within the U.S. state of West Virginia.
Sentence 2:  The company was founded by Kenney Grant in 1961.
  Fact 2.0: The company was founded by Kenney Grant.
  Fact 2.1: The company was founded in 1961.

Retrieved items:
<item 1>
Gino's Hamburgers
Gino's Hamburgers was a fast-food restaurant chain founded in Baltimore, Maryland, by Baltimore Colts defensive end Gino Marchetti and running back Alan Ameche,
along with their close friend Louis Fischer, in 1957.

 A new group of restaurants under the Gino's name involving some of the principals of the original chain was started in 2010.
</item 1>

<item 2>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 2>

<item 3>
Papa Gino's
Papa Gino's, Inc. is a restaurant chain based in Dedham, Massachusetts specializing in American-style pizza along with pasta, subs, salads, and a variety of
appetizers.

 There are over 150 Papa Gino's locations in Connecticut, Maine, Massachusetts, New Hampshire, and Rhode Island.
</item 3>

<item 4>
Graph query row:
organization: Gino's Pizza and Spaghetti
founded_on: 1961
</item 4>

<item 5>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 5>

<item 6>
Graph query row:
organization: Papa Murphy's
founded_on: 1995
</item 6>

<item 7>
Gino's East
Gino's East is a Chicago-based restaurant chain, notable for its deep-dish pizza (sometimes called Chicago-style pizza), and for its interior walls, which
patrons have covered in graffiti and etchings.

 The restaurant features deep-dish pizza baked in cast-iron pans, as well as sandwiches, soups and salads.
</item 7>

<item 8>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 8>

<item 9>
Papa Murphy's
Papa Murphy's, a business based in Vancouver, Washington, United States, is a take-and-bake pizza company.

 It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).

 The company and its franchisees operate more than 1,300 outlets in the United States and Canada.

 Papa Murphy's is the fifth-largest pizza chain in the United States.
</item 9>

<item 10>
Gino's Pizza and Spaghetti
Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.

 The company was founded by Kenney Grant in 1961.

 Many locations are shared with Tudor's Biscuit World although the Gino's brand is exclusive to West Virginia.

 There is one located in Ohio, while there are stand alone Tudor's locations in eastern Kentucky, southern Ohio and southwest Virginia.

 Gino's serves pizza, spaghetti, sandwiches, and more.

 Company headquarters are located in Huntington, West Virginia and Nitro, West Virginia.
</item 10>
=== END ITEM id=a33a0e6a7520168e ===

=== ITEM id=4ad3fc8d5e891ed9 ===
Gold sentences and facts:
Sentence 0: US Airways Flight 1549 was an Airbus A320-214 which, three minutes after takeoff from New York City's LaGuardia Airport on January 15, 2009, struck a flock of Canada geese just northeast of the George Washington Bridge and consequently lost all engine power.
  Fact 0.0: US Airways Flight 1549 was an Airbus A320-214.
  Fact 0.1: US Airways Flight 1549 took off from New York City's LaGuardia Airport.
  Fact 0.2: US Airways Flight 1549 took off on January 15, 2009.
  Fact 0.3: US Airways Flight 1549 struck a flock of Canada geese three minutes after takeoff.
  Fact 0.4: The bird strike happened just northeast of the George Washington Bridge.
  Fact 0.5: US Airways Flight 1549 consequently lost all engine power.
Sentence 1:  Unable to reach any airport, pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River off Midtown Manhattan.
  Fact 1.0: Pilots could not reach any airport.
  Fact 1.1: Pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River.
  Fact 1.2: The Hudson River ditching was off Midtown Manhattan.
Sentence 2: Chesley Burnett "Sully" Sullenberger III (born 1951) is an American retired airline captain celebrated for the January 15, 2009 water landing of US Airways Flight 1549 in the Hudson River off Manhattan after the plane was disabled by striking a flock of Canada geese immediately after takeoff; all 155 people aboard survived.
  Fact 2.0: Chesley Burnett "Sully" Sullenberger III was born in 1951.
  Fact 2.1: Chesley Sullenberger is an American retired airline captain.
  Fact 2.2: Chesley Sullenberger is celebrated for the January 15, 2009 water landing of US Airways Flight 1549 in the Hudson River off Manhattan.
  Fact 2.3: The plane was disabled by striking a flock of Canada geese immediately after takeoff.
  Fact 2.4: All 155 people aboard survived.

Retrieved items:
<item 1>
1965 Carmel mid-air collision
The 1965 Carmel mid-air collision occurred on December 4, 1965, when Eastern Air Lines Flight 853 (N6218C), a Lockheed Super Constellation en route from Boston
Logan International Airport to Newark International Airport, collided in mid-air with Trans World Airlines Flight 42 (N748TW), a Boeing 707-131B en route from
San Francisco International Airport to John F. Kennedy International Airport, over Carmel, New York, United States.
</item 1>

<item 2>
United Airlines Flight 736
United Airlines Flight 736 was a daily U.S. transcontinental passenger flight operated by United Airlines that crashed on April21, 1958, following a mid-air
collision.

 The aircraft assigned to Flight 736, a Douglas DC-7 airliner carrying 47 persons, was flying at cruise altitude above Clark County, Nevada, en route to a
stopover at Denver, Colorado, when it was struck by a United States Air Force fighter jet crewed by two pilots.

 The collision occurred at 8:30 a.m. in clear weather within a major commercial airway; both aircraft fell out of control from 21000 ft and crashed into
unpopulated desert terrain southwest of Las Vegas, Nevada.
</item 2>

<item 3>
Bird strike
A bird strike—sometimes called birdstrike, bird ingestion (for an engine), bird hit, or bird aircraft strike hazard (BASH)—is a collision between an airborne
animal (usually a bird or bat) and a manmade vehicle, especially an aircraft.

 The term is also used for bird deaths resulting from collisions with structures such as power lines, towers and wind turbines (see Bird-skyscraper collisions
and Towerkill).
</item 3>

<item 4>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 4>

<item 5>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 5>

<item 6>
Water landing
A water landing is, in the broadest sense, a landing on a body of water.

 Some aircraft such as floatplanes land on water as a matter of course.
</item 6>

<item 7>
The entries cover major aviation accidents, airlines, airports, and safety topics. Notable crashes include Flight 2311 (23 dead), Pan Am 806 (97), the 1948
Northwood collision (39), Tehran 1993 (133), Charkhi Dadri 1996 (349), and United 736 (49); Flight 1549 and the 1963 Neva ditching had no fatalities. TCAS warns
of collision threats, while bird strikes can cause crashes,
</item 7>

<item 8>
US Airways Flight 1549
US Airways Flight 1549 was an Airbus A320-214 which, three minutes after takeoff from New York City's LaGuardia Airport on January 15, 2009, struck a flock of
Canada geese just northeast of the George Washington Bridge and consequently lost all engine power.

 Unable to reach any airport, pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River off Midtown Manhattan.

 All 155 people aboard were rescued by nearby boats and there were few serious injuries.
</item 8>

<item 9>
Chesley Sullenberger
Chesley Burnett "Sully" Sullenberger III (born 1951) is an American retired airline captain celebrated for the January 15, 2009 water landing of US Airways
Flight 1549 in the Hudson River off Manhattan after the plane was disabled by striking a flock of Canada geese immediately after takeoff; all 155 people aboard
survived.

 Sullenberger is an international speaker on airline safety and has helped develop new protocols for airline safety.

 He served as the co-chairman, along with First Officer Jeffrey Skiles, of the EAA's Young Eagles youth introduction-to-aviation program from 2009 to 2013.

 He retired from US Airways after 30 years as a commercial pilot on March 3, 2010.

 In May of the following year, Sullenberger was hired by CBS News as an Aviation and Safety Expert.
</item 9>

<item 10>
ALM Flight 980
ALM Antillean Airlines Flight 980 was a flight scheduled to fly from John F. Kennedy International Airport in New York City to Princess Juliana International
Airport in St. Maarten, Netherlands Antilles, on 2 May 1970.

 After several unsuccessful landing attempts, the aircraft's fuel was exhausted and it made a forced water landing (ditching) in the Caribbean Sea 48 km (30 mi)
off St. Croix, with 23 fatalities and 40 survivors.

 The accident is one of a small number of intentional water ditchings of jet airliners.
</item 10>
=== END ITEM id=4ad3fc8d5e891ed9 ===

=== ITEM id=11553622ececcb5f ===
Gold sentences and facts:
Sentence 0: Karl Hermann Frank (24 January 1898 – 22 May 1946) was a prominent Sudeten German Nazi official in Czechoslovakia prior to and during World War II and an SS-"Obergruppenführer".
  Fact 0.0: Karl Hermann Frank's life dates are 24 January 1898 to 22 May 1946.
  Fact 0.1: Karl Hermann Frank was a prominent Sudeten German Nazi official in Czechoslovakia.
  Fact 0.2: Karl Hermann Frank was a Nazi official in Czechoslovakia prior to World War II.
  Fact 0.3: Karl Hermann Frank was a Nazi official in Czechoslovakia during World War II.
  Fact 0.4: Karl Hermann Frank was an SS-Obergruppenführer.
Sentence 1:  Until April 1942, it was the highest commissioned SS rank, inferior only to "Reichsführer-SS" (Heinrich Himmler or RFSS, which was the internal SS-abbreviation for Himmler) Translated as "senior group leader", the rank of "Obergruppenführer" was senior to "Gruppenführer".
  Fact 1.0: Until April 1942, the Obergruppenführer rank was the highest commissioned SS rank.
  Fact 1.1: Until April 1942, the Obergruppenführer rank was inferior only to Reichsführer-SS.
  Fact 1.2: Reichsführer-SS was the SS rank held by Heinrich Himmler.
  Fact 1.3: RFSS was the internal SS abbreviation for Himmler.
  Fact 1.4: Obergruppenführer translates as senior group leader.
  Fact 1.5: The rank of Obergruppenführer was senior to Gruppenführer.

Retrieved items:
<item 1>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 1>

<item 2>
Obergruppenführer
Obergruppenführer (] , "senior group leader") was a Nazi Party paramilitary rank that was first created in 1932 as a rank of the "Sturmabteilung" (SA), and
adopted by the "Schutzstaffel" (SS) one year later.

 Until April 1942, it was the highest commissioned SS rank, inferior only to "Reichsführer-SS" (Heinrich Himmler or RFSS, which was the internal SS-abbreviation
for Himmler) Translated as "senior group leader", the rank of "Obergruppenführer" was senior to "Gruppenführer".

 A similarly named rank of "Untergruppenführer" existed in the SA from 1929 to 1930 and as a title until 1933.

 In April 1942, the new rank of "SS-Oberst-Gruppenführer" was created which was above "Obergruppenführer" and below "Reichsführer-SS".
</item 2>

<item 3>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 3>

<item 4>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 4>

<item 5>
Karl Hermann Frank
Karl Hermann Frank (24 January 1898 – 22 May 1946) was a prominent Sudeten German Nazi official in Czechoslovakia prior to and during World War II and an
SS-"Obergruppenführer".

 He was tried, convicted and executed after World War II for his role in organizing the massacres of the people of the Czech villages of Lidice and Ležáky.
</item 5>

<item 6>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 6>

<item 7>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 7>

<item 8>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 8>

<item 9>
Frank A. Armstrong
Frank Alton Armstrong Jr. (May 24, 1902 – August 20, 1969) was a lieutenant general of the United States Air Force.

 As a brigadier general in the United States Army Air Forces during World War II, he was the inspiration for the main character in the novel and subsequent
film, "Twelve O'Clock High."

 After the war, he held a variety of senior leadership positions prior to and following the establishment of the USAF as an independent service in 1947.

 Promoted to major general in 1950, he advanced to lieutenant general in 1956 and retired at that rank in 1962.
</item 9>

<item 10>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 10>
=== END ITEM id=11553622ececcb5f ===

=== ITEM id=df0e95cc8d90a845 ===
Gold sentences and facts:
Sentence 0: Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry, famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
  Fact 0.0: Aram Saroyan was born on September 25, 1943.
  Fact 0.1: Aram Saroyan is an American poet.
  Fact 0.2: Aram Saroyan is an American novelist.
  Fact 0.3: Aram Saroyan is an American biographer.
  Fact 0.4: Aram Saroyan is an American memoirist.
  Fact 0.5: Aram Saroyan is an American playwright.
  Fact 0.6: Aram Saroyan is especially known for his minimalist poetry.
  Fact 0.7: A famous example of Aram Saroyan's minimalist poetry is the one-word poem "lighght".
  Fact 0.8: A famous example of Aram Saroyan's minimalist poetry is a one-letter poem comprising a four-legged version of the letter "m".
Sentence 1: Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of microelectronics.
  Fact 1.0: Dr. Edward Keonjian was born on 14 August 1909.
  Fact 1.1: Dr. Edward Keonjian died on 6 September 1999.
  Fact 1.2: Dr. Edward Keonjian was a prominent engineer.
  Fact 1.3: Dr. Edward Keonjian was an early leader in the field of low-power electronics.
  Fact 1.4: Dr. Edward Keonjian was the father of microelectronics.

Retrieved items:
<item 1>
Edward Keonjian
Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of
microelectronics.

 In 1954 Keonjian designed the world's first solar-powered, pocket-sized radio transmitter.

 In 1959 Keonjian designed the first prototype of integrated circuit.

 In 1963 he organized the world's first international symposium on low-power electronics.

 Later on Keonjian collaborated with NASA astronaut Neil Armstrong as chief of failure analysis on the Apollo 11 project.
</item 1>

<item 2>
Edmund Mortimer (1302–1331)
Sir Edmund Mortimer (1302/1303 – 16 December 1331) was the eldest son of Roger Mortimer, 1st Earl of March and Joan de Geneville, 2nd Baroness Geneville.

 By his wife Elizabeth de Badlesmere he was the father of Roger Mortimer, 2nd Earl of March.

 Though Edmund survived his father by one year, he did not inherit his father's lands and titles as they were forfeited to the Crown and his son only reacquired
them gradually.
</item 2>

<item 3>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 3>

<item 4>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 4>

<item 5>
Strawberry Saroyan
Strawberry Saroyan (born 1970) is a journalist and author.

 The daughter of Aram Saroyan and granddaughter of playwright William Saroyan and actress Carol Matthau, she spent her childhood in Bolinas, California.

 She has a sister named Cream.

 She writes for the New York Times Style section and is the author of "".
</item 5>

<item 6>
Aram Saroyan
Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry,
famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
</item 6>

<item 7>
My Name Is Aram
My Name is Aram is a book of short stories by William Saroyan first published in 1940.

 The stories detail the exploits of Aram Garoghlanian, a boy of Armenian descent growing up in Fresno, California, and the various members of his large family.

 This book is assigned reading in some schools.
</item 7>

<item 8>
Edmund Mortimer, 5th Earl of March
Edmund de Mortimer, 5th Earl of March and 7th Earl of Ulster (6 November 1391 – 18 January 1425) was an English nobleman.

 A great-great-grandson of King Edward III of England, he was heir presumptive to King Richard II of England, his first cousin twice removed, when Richard II
was deposed in favour of Henry IV.

 Edmund Mortimer's claim to the throne was the basis of rebellions and plots against Henry IV and his son Henry V, and was later taken up by the House of York
in the Wars of the Roses, though Mortimer himself was an important and loyal vassal of Henry V and Henry VI.

 Edmund Mortimer was the last Earl of March of the Mortimer family.
</item 8>

<item 9>
Levan of Imereti
Levan (Georgian: ლევანი ) (1573–1590), of the Bagrationi Dynasty, was a king of Imereti from 1585 to 1588.

 He succeeded on the death of his father, George II, in 1585 when he was twelve years old.

 With his ascend to the throne, Leon faced a revolt by his own uncle, Constantine, who defied the royal authority and took control of Upper Imereti.

 Leon made an alliance with the Mingrelian prince Mamia IV Dadiani, married his sister Marekhi, and forced Constantine to surrender in 1587.

 A year later, Imereti was invaded by Simon I, the resurgent king of Kartli in eastern Georgia, who sought to reunify all Georgian lands under his crown.

 Leon was forced to flee to the highland province of Lechkhumi, but was soon able to resume the throne after Simon had to return to Kartli.
</item 9>

<item 10>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 10>
=== END ITEM id=df0e95cc8d90a845 ===

=== ITEM id=68fe994af99e4f66 ===
Gold sentences and facts:
Sentence 0: Inner Secrets is the ninth studio album by Santana.
  Fact 0.0: Inner Secrets is a studio album by Santana.
  Fact 0.1: Inner Secrets is Santana's ninth studio album.
Sentence 1: Carlos Santana    (born July 20, 1947) is a Mexican and American musician who first became famous in the late 1960s and early 1970s with his band, Santana, which pioneered a fusion of rock and Latin American music.
  Fact 1.0: Carlos Santana was born on July 20, 1947.
  Fact 1.1: Carlos Santana is a Mexican musician.
  Fact 1.2: Carlos Santana is an American musician.
  Fact 1.3: Carlos Santana first became famous in the late 1960s and early 1970s.
  Fact 1.4: Carlos Santana first became famous with his band Santana.
  Fact 1.5: The band Santana pioneered a fusion of rock and Latin American music.

Retrieved items:
<item 1>
Graph query row:
entities: ['Santana', 'Inner Secrets']
relationships: ['CREATED_BY']
</item 1>

<item 2>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 2>

<item 3>
Information (Berlin album)
Information is the first album by Berlin, released in 1980 by Vinyl Records.

 It was recorded during the period when Terri Nunn had temporarily left the band to pursue an acting career and Virginia Macolino performed the lead vocals.

 Several songs were written by previous lead singer Toni Childs who went solo prior to the album's release.
</item 3>

<item 4>
Inner Secrets
Inner Secrets is the ninth studio album by Santana.

 It was released in 1978 and marks the start of the phase of Santana's career where he moved away from the fusion of Latin, jazz, rock and blues that marked his
previous records and began to move towards an album-oriented rock direction.

 As such, the album's quality is widely disputed among fans. "

Stormy" and "One Chain (Don't Make No Prison)" were both hit singles.

 In The Netherlands "Well All Right" was released as a single and reached #22 in the top 40.
</item 4>

<item 5>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 5>

<item 6>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 6>

<item 7>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 7>

<item 8>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 8>

<item 9>
Chicago 13
Chicago 13 is the eleventh studio album by the American band Chicago, released in 1979.

 The follow-up to "Hot Streets", "Chicago 13" is often critically disfavored.

 This would be the band's final release that features lead guitarist Donnie Dacus, who had followed late, founding lead guitarist, Terry Kath.

 All band members would contribute to the songwriting (one of only two albums where this is the case, with the other being "Chicago VII").
</item 9>

<item 10>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 10>
=== END ITEM id=68fe994af99e4f66 ===

=== ITEM id=5fa94ead2b2b4524 ===
Gold sentences and facts:
Sentence 0: The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver Division, the Shenango Division, and the Conneaut Division.
  Fact 0.0: The Beaver and Erie Canal was also known as the Erie Extension Canal.
  Fact 0.1: The Beaver and Erie Canal was part of the Pennsylvania Canal system.
  Fact 0.2: The Beaver and Erie Canal consisted of three sections.
  Fact 0.3: The three sections of the Beaver and Erie Canal were the Beaver Division, the Shenango Division, and the Conneaut Division.
Sentence 1: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.
  Fact 1.0: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp.
  Fact 1.1: The Dismal Swamp Canal is located in Virginia and North Carolina.
  Fact 1.2: Virginia and North Carolina are in the United States.

Retrieved items:
<item 1>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 1>

<item 2>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 2>

<item 3>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 3>

<item 4>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 4>

<item 5>
Washington City Canal
The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at
that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).

 The canal fell into disuse in the late 19th century and the city government covered over or filled in various sections.
</item 5>

<item 6>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 6>

<item 7>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 7>

<item 8>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 8>

<item 9>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 9>

<item 10>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 10>
=== END ITEM id=5fa94ead2b2b4524 ===

=== ITEM id=1204a61a0fbfc2f7 ===
Gold sentences and facts:
Sentence 0:  At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
  Fact 0.0: She won a silver medal at the 2002 Commonwealth Games.
  Fact 0.1: She won the silver medal with Sara Petersen in the women's doubles at the 2002 Commonwealth Games.
  Fact 0.2: She won a bronze medal in the mixed team at the 2002 Commonwealth Games.
Sentence 1: The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25 July to 4 August 2002.
  Fact 1.0: The 2002 Commonwealth Games were officially known as the XVII Commonwealth Games.
  Fact 1.1: The 2002 Commonwealth Games were commonly known as Manchester 2002.
  Fact 1.2: The 2002 Commonwealth Games were held in Manchester, England.
  Fact 1.3: The 2002 Commonwealth Games were held from 25 July to 4 August 2002.

Retrieved items:
<item 1>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 1>

<item 2>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 2>

<item 3>
Nicole Gordon (badminton)
Nicole Gordon (born 17 March 1976 in Lower Hutt, New Zealand) is a female badminton player from New Zealand.

 At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
</item 3>

<item 4>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 4>

<item 5>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 5>

<item 6>
Graph query row:
person: Nicole Gordon
relationship: None
event: 2002 Commonwealth Games
host_city: Manchester
</item 6>

<item 7>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 7>

<item 8>
2010 US Open – Mixed Doubles
Carly Gullickson and Travis Parrott were the defending champions, but lost in the first round to Gisela Dulko and Pablo Cuevas 2–6, 4–6.
</item 8>

<item 9>
1999 Copa Colsanitas – Doubles
The 1999 Copa Colsanitas – Doubles was the doubles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Janette Husárová and Paola Suárez were the defending champions but only Suárez competed that year with Laura Montalvo.
</item 9>

<item 10>
Nicole Strausak
Nicole Strausak (born October 21, 1971) is a Swiss curler who retired from competitive curling in March 2005.

 Amongst other awards, she received a World Junior Silver medal in 1991, a European Silver medal in 1993 and a World Silver medal in 2000.

 In 1999, at the European Curling Championships in Chamonix, as well as receiving a Bronze medal, she was awarded the Fair Play & Friendly Award by her fellow
athletes for her conduct and attitude, on and off the ice.
</item 10>
=== END ITEM id=1204a61a0fbfc2f7 ===

=== ITEM id=0875e78046131e1f ===
Gold sentences and facts:
Sentence 0: The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.
  Fact 0.0: The Bishop of Ely is the ordinary of the Diocese of Ely.
  Fact 0.1: The Diocese of Ely is a Church of England diocese.
  Fact 0.2: The Diocese of Ely is in the Province of Canterbury.
Sentence 1:  Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
  Fact 1.0: Conway became Bishop of Ely in 2010.
  Fact 1.1: Conway was translated to Bishop of Ely from the Diocese of Salisbury.
  Fact 1.2: Conway was Bishop suffragan of Ramsbury before becoming Bishop of Ely.
  Fact 1.3: Conway was Bishop suffragan of Ramsbury in the Diocese of Salisbury.
Sentence 2: The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.
  Fact 2.0: The Diocese of Salisbury is a Church of England diocese.
  Fact 2.1: The Diocese of Salisbury is in the south of England.
  Fact 2.2: The Diocese of Salisbury is within the ecclesiastical Province of Canterbury.

Retrieved items:
<item 1>
Ecclesiastical province
An ecclesiastical province is a general term for one of the basic forms of jurisdiction in Christian Churches with traditional hierarchical structure, including
Western Christianity and Eastern Christianity.

 In general, ecclesiastical province is consisted of several dioceses (or eparchies), one of them being the archdiocese (or archeparchy), headed by metropolitan
bishop or archbishop who has ecclesiastical jurisdiction over all other bishops of the province.
</item 1>

<item 2>
Dean of Chelmsford
The Dean of Chelmsford is the head ("primus inter pares" – first among equals) and chair of the Cathedral Chapter, the governing body of Chelmsford Cathedral,
the Cathedral Church of St Mary the Virgin, St Peter and St Cedd.

 Before 2000 the post was designated as a provost, which was then the equivalent of a dean at most English cathedrals.

 The cathedral is the mother church of the Diocese of Chelmsford and seat of the Bishop of Chelmsford.

 The Dean of Chelmsford is also responsible for the Chapel of St Peter-on-the-Wall at Bradwell-on-Sea, founded by St Cedd, among the oldest church buildings in
regular use in England.
</item 2>

<item 3>
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 3>

<item 4>
Ecclesiastical Province of Freiburg
The Ecclesiastical Province of Freiburg ("Kirchenprovinz Freiburg") or Upper Rhenish Ecclesiastical Province ("Oberrheinische Kirchenprovinz") is an
ecclesiastical province of the Roman Catholic Church in the Upper Rhine area of Germany, centring on Freiburg im Breisgau.

 It covers the Roman Catholic Archdiocese of Freiburg, the Roman Catholic Diocese of Mainz and the Roman Catholic Diocese of Rottenburg-Stuttgart, covering
large areas of Baden-Württemberg and Hesse and small parts of Rhineland-Palatinate.

 Its metropolitan bishop is the Archbishop of Freiburg - that Archdiocese and the Province were both set up in 1821 in the wake of the 1801 Concordat and the
1815 Congress of Vienna.
</item 4>

<item 5>
Roman Catholic Ecclesiastical Province of Miami
The Ecclesiastical Province of Miami is a Catholic ecclesiastical province covering the U.S. state of Florida.

 Its metropolitan bishop is the Archbishop of Miami, head of the Archdiocese of Miami.

 The province additionally includes the suffragan dioceses of Orlando, Palm Beach, Pensacola-Tallahassee, St. Augustine, St. Petersburg, and Venice.
</item 5>

<item 6>
Ecclesiastical Province of Ontario
The Ecclesiastical Province of Ontario is one of the Anglican Church of Canada's four ecclesiastical provinces.

 It was established in 1912 out of six dioceses of the Ecclesiastical Province of Canada located in the civil Province of Ontario, and the Diocese of Moosonee
from the Ecclesiastical Province of Rupert's Land.
</item 6>

<item 7>
Diocese of Salisbury
The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.

 The diocese covers most of Dorset (excepting the deaneries of Bournemouth and Christchurch, which fall within the Diocese of Winchester), and most of Wiltshire
(excepting a part in the north and Swindon).

 The diocese is led by the Bishop of Salisbury (Nick Holtam) and the diocesan synod.

 The bishop's seat is at Salisbury Cathedral.
</item 7>

<item 8>
Roman Catholic Archdiocese of Cashel and Emly
The Archdiocese of Cashel and Emly (Irish: "Ard-Deoise Chaisil agus Imligh" ) is a Roman Catholic archdiocese in mid-western Ireland.

 The archdiocese is led by the Archbishop of Cashel and Emly, who serves as pastor of the mother church, the Cathedral of the Assumption and metropolitan of the
ecclesiastical province of Cashel and Emly.

 The Diocese of Cashel was established in 1111 by the Synod of Rathbreasail.

 The ecclesiastical province, which was roughly co-extensive with the secular province of Munster, was created in 1152 by the Synod of Kells.

 The cathedral church of the archdiocese is located in Thurles, County Tipperary.

 The incumbent archbishop is Kieran O'Reilly.
</item 8>

<item 9>
Dean of Guildford
The Dean of Guildford is the head ("primus inter pares" – first among equals) and chair of the chapter of canons, the ruling body of Guildford Cathedral.

 The dean and chapter are based at the "Cathedral Church of the Holy Spirit" in Guildford.

 The cathedral is the mother church of the Diocese of Guildford and seat of the Bishop of Guildford.

 The current dean is Dianna Gwilliams, who was installed on 15 September 2013.
</item 9>

<item 10>
Bishop of Ely
The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.

 The diocese roughly covers the county of Cambridgeshire (with the exception of the Soke of Peterborough), together with a section of north-west Norfolk and has
its episcopal see in the City of Ely, Cambridgeshire, where the seat is located at the Cathedral Church of the Holy Trinity.

 The current bishop is Stephen Conway, who signs "+Stephen Elien:" (abbreviation of the Latin adjective "Eliensis", meaning "of Ely").

 The diocesan bishops now reside in Bishop's House, Ely, the former cathedral deanery.

 Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
</item 10>
=== END ITEM id=0875e78046131e1f ===
