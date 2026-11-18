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

=== ITEM id=7e3d1114cde4a13b ===
Gold sentences and facts:
Sentence 0: Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.
  Fact 0.0: Victor Ronald Salva was born March 29, 1958.
  Fact 0.1: Victor Ronald Salva is an American film director.
  Fact 0.2: Victor Ronald Salva is an American screenwriter.
Sentence 1: Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.
  Fact 1.0: Emilio "El Indio" Fernández was born Emilio Fernández Romo.
  Fact 1.1: Emilio "El Indio" Fernández was born March 26, 1904.
  Fact 1.2: Emilio "El Indio" Fernández died August 6, 1986.
  Fact 1.3: Emilio "El Indio" Fernández was a Mexican film director.
  Fact 1.4: Emilio "El Indio" Fernández was a Mexican actor.
  Fact 1.5: Emilio "El Indio" Fernández was a Mexican screenwriter.

Retrieved items:
<item 1>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 1>

<item 2>
Columba Domínguez
Columba Domínguez Adalid (March 4, 1929 – August 13, 2014) was a Mexican actress, singer, and painter.

 Considered a crucial figure in the Golden Age of Mexican cinema, she was one of the muses of the film director Emilio Fernández, with whom she was romantically
linked for several years.

 She is remembered particularly for her performance in the film "Pueblerina" (1949), which is considered one of the jewels of the Mexican Cinema.
</item 2>

<item 3>
José Antonio Santamaría Mikel Vaqueriza
José Antonio Santamaría Mikel Vaqueriza (born 16 March 1946 in San Sebastián, Guipúzcoa, Spain; died 19 January 1993) was a Spanish footballer.

 During his career as a professional footballer, Santamaría played for SD Eibar (1963 to 1964), Real Sociedad (1964 to 1967), Real Sociedad (1967 to 1971),
Hercules CF (1971 to 1974) finishing his career at CE Sabadell FC (1974 to 1975).

 After retiring from football he had a career as an entrepreneur in the hospitality sector.

 He was killed in an attack committed by ETA in 1993.
</item 3>

<item 4>
Comedy in the Golden Age of Mexican cinema
Sadly, it was not as successful as many directors and film maker though it would be.

 Many failed to correctly synchronize the audio with the picture.

 Although to some extent it may have been unsuccessful, it is agreeable that it helped revolutionize film in Latin America.

 Mexico’s film industry takes its height post “El Cine Sonoro” to what is known as “The Golden Age”.

 Many film members consider this one of the most successful eras in the history of Mexican cinema.

 Many recognize the films of this era when they see black and white.

 It is also important to note that the films also portrayed social issues.

 Some of the most influential directors and actors are Emilio Fernández, Raul De Anda, Roberto B. Cervenna, Cantinflas, Vitola, Tin Tan, etc.

 The focus of this article will be on some of top grossing comedian actors of the time Cantinflas, Tin Tan, Clavillazo and others.
</item 4>

<item 5>
Antonio De Carlo
Antonio De Carlo, born (1967--) 4, 1967 (age 50 ) in Tijuana, Baja California, Mexico, is a Mexican actor also known as Antonio in the late 1980s and early
1990s.

 He won an Emmy Award in 2005-2006.

 He is president and founder of the "Fundación Cultura Sin Fronteras AC".

 After twelve years out of show business, in 2012 he returned to the soap operas performing "Magic Dragon" one of the main characters for "Miss XV", a
co-production from Televisa México and Nickelodeon directed and produced by Pedro Damian.

 He is also part of the cast of "Muchacha Italiana Viene a Casarse", directed by Pedro Damian (Televisa, México).
</item 5>

<item 6>
Emilio Fernández
Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.

 He was one of the most prolific film directors of the Golden Age of Mexican cinema in the 1940s and 1950s.

 He is best known for his work as director of the film "Maria Candelaria" (1944), which won the Palme d'Or award at the 1946 Cannes Film Festival.

 As an actor, he worked in numerous film productions in Mexico and in Hollywood.
</item 6>

<item 7>
Valerio Fioravanti
Giuseppe Valerio "Giusva" Fioravanti (born 28 March 1958) is an Italian who, with Francesca Mambro, was a leading figure in a far right terrorist group "Nuclei
Armati Rivoluzionari" ("Armed Revolutionary Nuclei" or NAR).

 Fioravanti appeared in films and television from a young age, and in his early teens was the most famous child in Italy.

 He and Mambro were fugitives wanted for terrorist offences by their early twenties, they spent a further period on the run while wanted for the Bologna
bombing.

 Both were captured after gunfights with police.

 Although Fioravanti, Mambro and a third NAR mamber were convicted of the bombing, there were those who thought that a higher level of the conspiracy was never
uncovered.

 Fioravanti and Mambro admit carrying out terrorist murders, but have always denied any involvement in the Bologna station bomb.
</item 7>

<item 8>
Fernando Fernández (actor)
Fernando Fernández "El Crooner de México" (1916–1999) was a Mexican actor and singer.

 He was born on November 9, 1916, in Monterrey, Mexico, and died in Mexico City, Mexico, on November 24, 1999.

 Fernando was the son of Eloisa Reyes.

 He was brother of the famous film director Emilio Fernández and actor Jaime Fernández.

 He was married to singer Lupita Palomera, who died in 2008.

 Fernando Fernández was known as "the Crooner of Mexico."
</item 8>

<item 9>
Jaime Fernández (actor)
Jaime Fernández Reyes (December 6, 1937 – April 15, 2005) was a Mexican actor.

 Over his career, he won 3 Silver Ariel awards - the Mexican equivalent of the Oscar - including one for what is arguably his best-known role, playing Friday in
Luis Buñuel's "Robinson Crusoe".

 He appeared in over 200 films and served as the General Secretary of the Mexican actors' union for 11 years.

 His older brother was actor/director Emilio Fernández.
</item 9>

<item 10>
Filiberto Ojeda Ríos
Filiberto Ojeda Ríos (April 26, 1933 – September 23, 2005) was the commander-in-chief ("Responsable General") of the Boricua Popular Army ("Ejército Popular
Boricua, a.k.a., Los Macheteros").

 According to an unsigned article in the "Los Angeles Times", Los Macheteros was “a terrorist group seeking Puerto Rico's independence.”

 The group campaigned for, and supported, the independence of Puerto Rico from the United States.

 In 2001, FBI Director Louis J. Freeh claimed the group was linked to acts of terrorism, but Ronald Fernandez, scholar of Puerto Rican history, suggests such
labeling was an act of political convenience by the United States Government, intended to "shift the blame for any attacks on U.S. policy or personnel from us
to them".

 Ríos was also a founder of the FALN.

 In a 1983 New York Times article, Robert McFadden described the FALN as a Puerto Rican terrorist organization responsible for bombings during the 1970s and
early 1980s "in the name of Puerto Rican independence".
</item 10>
=== END ITEM id=7e3d1114cde4a13b ===

=== ITEM id=9c8d22dc472b0de9 ===
Gold sentences and facts:
Sentence 0:  They competed at the 2015 Pacific Games, where they finished with an 0-4 record.
  Fact 0.0: The unnamed subject competed at the 2015 Pacific Games
  Fact 0.1: The unnamed subject finished the 2015 Pacific Games with an 0-4 record
Sentence 1: Basketball at the 2015 Pacific Games in Port Moresby, Papua New Guinea was held at the BSP Arena and PNG Power Dome on 3–12 July 2015.
  Fact 1.0: Basketball at the 2015 Pacific Games was held in Port Moresby, Papua New Guinea
  Fact 1.1: Basketball at the 2015 Pacific Games was held at the BSP Arena and PNG Power Dome
  Fact 1.2: Basketball at the 2015 Pacific Games was held on 3 to 12 July 2015

Retrieved items:
<item 1>
Vanuatu national under-23 football team
The Vanuatu national under-23 football team, also known as Vanuatu Cyclone, represents Vanuatu at U23 tournaments.

 The team is considered to be the feeder team for the Vanuatu national football team.

 They are controlled by the Vanuatu Football Federation.

 The team has gained notoriety for thrashing Micronesia's side 46–0 in the 2015 Pacific Games.
</item 1>

<item 2>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 2>

<item 3>
Kiribati national basketball team
The Kiribati national basketball team are the basketball side that represent Kiribati in international competitions.

 They competed at the 2015 Pacific Games, where they finished with an 0-4 record.
</item 3>

<item 4>
2015 Pacific Games
The 2015 Pacific Games was held in Port Moresby, Papua New Guinea, from 4 to 18 July 2015.

 It was the fifteenth staging of the Pacific Games.
</item 4>

<item 5>
Powerlifting at the 2015 Pacific Games
Powerlifting at the 2015 Pacific Games was held from 9–10 July 2015 in the Sir John Guise Indoor Power Dome at Port Moresby, Papua New Guinea.

 The host nation Papua New Guinea was the dominant team, particularly in the women's divisions where it claimed six of the seven possible gold medals.

 Nauru was the strongest team in the men's divisions, winning four of the eight weight categories.

 Telupe Iosefa received the first ever gold medal won by Tuvalu at the Pacific Games in the 120 kg male division.
</item 5>

<item 6>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 6>

<item 7>
Nicolas N'Godrela
Nicolas N'Godrela (born (6 October 1984) is a New Caledonian tennis player.

 He won the Men's singles and Men's doubles gold medal at the 2015 Pacific Games.

 Apart from that, he was also part of the New Caledonian Tennis Team which won Gold at the 2015 Pacific Games Men's Team Event.
</item 7>

<item 8>
Men's football at the 2015 Pacific Games
The 2015 Pacific Games men's football tournament was the 14th edition of the Pacific Games men's football tournament.

 The men's football tournament was held in Port Moresby, Papua New Guinea between 3–17 July 2015 as part of the 2015 Pacific Games.

 The tournament was age-restricted and open to men's under-23 national teams only.
</item 8>

<item 9>
Kiribati at the 2015 Pacific Games
Kiribati competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 A total of 86 competitors for Kiribati were listed as of 4 July 2015.
</item 9>

<item 10>
Basketball at the 2015 Pacific Games
Basketball at the 2015 Pacific Games in Port Moresby, Papua New Guinea was held at the BSP Arena and PNG Power Dome on 3–12 July 2015.
</item 10>
=== END ITEM id=9c8d22dc472b0de9 ===

=== ITEM id=eeb792b4c5a5b760 ===
Gold sentences and facts:
Sentence 0: Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.
  Fact 0.0: Kenneth William Kwapis, also known as Ken Kwapis, was born on August 17, 1957.
  Fact 0.1: Ken Kwapis is an American film director.
  Fact 0.2: Ken Kwapis is an American television director.
  Fact 0.3: Ken Kwapis is an American screenwriter.
Sentence 1:  He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
  Fact 1.0: He specialized in the single-camera sitcom in the 1990s.
  Fact 1.1: He specialized in the single-camera sitcom in the 2000s.
  Fact 1.2: He has directed feature films.
  Fact 1.3: He directed Sesame Street Presents Follow That Bird, released in 1985.
  Fact 1.4: He directed The Sisterhood of the Traveling Pants, released in 2005.
  Fact 1.5: He directed He's Just Not That into You, released in 2009.
Sentence 2:  As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).
  Fact 2.0: As a filmmaker, he has directed Valle de Lágrimas, released in 2006.
  Fact 2.1: As a filmmaker, he has directed Peace Grows, released in 2004.

Retrieved items:
<item 1>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 1>

<item 2>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 2>

<item 3>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 3>

<item 4>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 4>

<item 5>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 5>

<item 6>
The Private Eyes (1976 film)
The Private Eyes is a 1976 Hong Kong comedy film written, directed by and starring Michael Hui and co-starring his brothers, Samuel Hui and Ricky Hui, and also
starring Shih Kien and Richard Ng in his second film role.

 John Woo was the production designer and also co-director though he was uncredited.

 Sammo Hung served as the film's action director and Jackie Chan was also a stuntman.

 This is the third film of the Hui Brothers and it is the first film that established the Hui Brothers' comedies internationally.
</item 6>

<item 7>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 7>

<item 8>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 8>

<item 9>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 9>

<item 10>
Jon Paul Puno
Jon Paul Puno (born 1985) is an American filmmaker and classical crossover singer.

 As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).

 And also played a small role in the 2006 movie I Will Always Love You (film).
</item 10>
=== END ITEM id=eeb792b4c5a5b760 ===

=== ITEM id=8f67332422653a7c ===
Gold sentences and facts:
Sentence 0: Mirpur University of Science & Technology (میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی) (MUST) was formerly a constituent college of University of Azad Jammu and Kashmir as University College of Engineering & Technology Mirpur (UCET Mirpur).
  Fact 0.0: Mirpur University of Science & Technology (MUST) was formerly a constituent college of University of Azad Jammu and Kashmir.
  Fact 0.1: The Urdu name of Mirpur University of Science & Technology is میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی.
  Fact 0.2: Mirpur University of Science & Technology was formerly named University College of Engineering & Technology Mirpur (UCET Mirpur).
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen is a university located in Debrecen, Hungary.
  Fact 1.1: The Hungarian name of the University of Debrecen is 'Debreceni Egyetem'.

Retrieved items:
<item 1>
Graph query row:
university: University of Debrecen
university_status: university
location: Debrecen
country: Hungary
</item 1>

<item 2>
Albanian University in Berat
Through presentations, research and exchange, the University directs and encourages the identification and development of projects that will once again
consolidate its primary purpose, scientific development, and research toward better quality.
</item 2>

<item 3>
Albanian University in Berat
The University of Berat is a private institution, independent and secular Higher Education, which operates the state fee.

 Licensed by the Council of Ministers, no. 697 dated 18.06.2009, he opened the doors in the academic year 2009- 2010, offering programs in two faculties:
Faculty of Social Sciences Programs in Science Teaching Legal, Political Science and Psychology General Administration and the Faculty of Economics with
Learning programs in Banking and Finance-Business Administration.

 During the first academic year, about 1,000 students attended studies.

 A year later, this new institution offering undergraduate degrees in study programs: Preschool Teacher Education, Primary Education Teacher, English Language,
Nursing, Engineering Computer and Information Technology, thus finalizing the 2010-2011 academic year with the opening of two new faculties: Faculty of Medical
Sciences and the Faculty of Architecture and Inxhinierive.

 Academic Year 2010-2011 finds the University Berat with about 2600 students in the first cycle of study.

 Since the first year of its inception, the institution pointed at the center of academic life, the promotion of intellectual product and melting learning and
research.
</item 3>

<item 4>
Mirpur University of Science and Technology
Mirpur University of Science & Technology (میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی) (MUST) was formerly a constituent college of University of Azad Jammu and
Kashmir as University College of Engineering & Technology Mirpur (UCET Mirpur).

 It is a state university and the President of Azad Jammu & Kashmir is the Chancellor of the university.

 The Vice-Chancellor is the executive head and manages the university functions.
</item 4>

<item 5>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 5>

<item 6>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 6>

<item 7>
Graph query row:
university: Mirpur University of Science and Technology
university_status: state university
location: None
country: None
</item 7>

<item 8>
Hamdard University
Hamdard University (Urdu: ) is a private research university with campuses in Karachi and Islamabad, Pakistan.

 It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.

 Hamdard is one of the first and the oldest private institutions of higher education in Pakistan.

 In Karachi, Hamdard University is the largest private research university with a campus area of over 350 acres.
</item 8>

<item 9>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 9>

<item 10>
Bangladesh University of Business and Technology
Bangladesh University of Business and Technology (Bengali: বাংলাদেশ ইউনিভার্সিটি অফ বিজনেস অ্যান্ড টেকনোলজি ) or BUBT is one of the eight private universities
to receive the green Signal from the government, located in Mirpur, Dhaka, Bangladesh.

 The university was established under the Private University Act 1992.

 BUBT is regulated by the Bangladesh University Grants Commission (UGC).
</item 10>
=== END ITEM id=8f67332422653a7c ===

=== ITEM id=ea8432a1ace70b0e ===
Gold sentences and facts:
Sentence 0: Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).
  Fact 0.0: Blinding Edge Pictures is an American film production company.
  Fact 0.1: Blinding Edge Pictures was founded in 2000 by M. Night Shyamalan.
  Fact 0.2: Blinding Edge Pictures is known for producing films written and directed by Shyamalan.
  Fact 0.3: Blinding Edge Pictures produced Unbreakable (2000), written and directed by Shyamalan.
  Fact 0.4: Blinding Edge Pictures produced Signs (2002), written and directed by Shyamalan.
  Fact 0.5: Blinding Edge Pictures produced The Village (2004), written and directed by Shyamalan.
  Fact 0.6: Blinding Edge Pictures produced The Happening (2008), written and directed by Shyamalan.
  Fact 0.7: Blinding Edge Pictures produced After Earth (2013), written and directed by Shyamalan.
  Fact 0.8: Blinding Edge Pictures produced The Visit (2015), written and directed by Shyamalan.
  Fact 0.9: Blinding Edge Pictures produced Split (2017), written and directed by Shyamalan.
Sentence 1: Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson, alongside Robin Wright and Spencer Treat Clark.
  Fact 1.0: Unbreakable is a 2000 American superhero thriller film.
  Fact 1.1: Unbreakable was written by M. Night Shyamalan.
  Fact 1.2: Unbreakable was produced by M. Night Shyamalan.
  Fact 1.3: Unbreakable was directed by M. Night Shyamalan.
  Fact 1.4: Unbreakable stars Bruce Willis.
  Fact 1.5: Unbreakable stars Samuel L. Jackson.
  Fact 1.6: Robin Wright is in the cast of Unbreakable.
  Fact 1.7: Spencer Treat Clark is in the cast of Unbreakable.

Retrieved items:
<item 1>
Cop Out (2010 film)
Cop Out is a 2010 American buddy cop action-comedy film directed and edited by Kevin Smith, written by Mark and Robb Cullen and starring Bruce Willis, Tracy
Morgan, Kevin Pollak and Seann William Scott.

 The plot revolves around two veteran NYPD partners (Willis and Morgan) on the trail of a stolen, rare, mint-condition baseball card who find themselves up
against a relentless, memorabilia-obsessed bloodthirsty gangster.

 This is the first film that Smith directed that he did not also write.

 Upon its release, the film was met with negative reviews by critics and underperformed at the box office.

 It is nevertheless Kevin Smith's highest grossing film.
</item 1>

<item 2>
Graph query row:
film: Unbreakable
stars: []
</item 2>

<item 3>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 3>

<item 4>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 4>

<item 5>
Last Man Standing (1996 film)
Last Man Standing is a 1996 American action thriller film written and directed by Walter Hill and starring Bruce Willis, Christopher Walken and Bruce Dern.

 It is a credited remake of Akira Kurosawa's "Yojimbo".
</item 5>

<item 6>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 6>

<item 7>
Unbreakable (film)
Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson,
alongside Robin Wright and Spencer Treat Clark.

 The movie is the first installment in a trilogy.

 In "Unbreakable", a security guard named David Dunn survives a horrific train crash.

 After the incident, with the help of a manipulative disabled comic book shop owner named Elijah Price, he learns that he possesses superhuman powers.

 As Dunn explores and reluctantly confronts his powers while trying to navigate a difficult family life, he begins to fight crime and learns the true nature of
Elijah Price.
</item 7>

<item 8>
Blinding Edge Pictures
Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by
Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).

 In 2015, the company released its first television series "Wayward Pines".
</item 8>

<item 9>
Striking Distance
Striking Distance is a 1993 American action thriller film starring Bruce Willis as Pittsburgh Police homicide detective Thomas Hardy.

 The film co-stars Sarah Jessica Parker, Dennis Farina, and Tom Sizemore.

 It was directed by Rowdy Herrington and written by Herrington and Marty Kaplan.

 The film was shot on location throughout Pittsburgh; its early title was "Three Rivers".
</item 9>

<item 10>
Mercury Rising
Mercury Rising is a 1998 American political action thriller film starring Bruce Willis and Alec Baldwin.

 Directed by Harold Becker, the movie is based on Ryne Douglas Pearson's 1996 novel originally published as "Simple Simon".

 Willis plays Art Jeffries, an undercover FBI agent who protects a 9-year-old boy with autism who is targeted by government assassins after he cracks a top
secret government code.
</item 10>
=== END ITEM id=ea8432a1ace70b0e ===

=== ITEM id=4ccd80e88ae82448 ===
Gold sentences and facts:
Sentence 0: The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic Beach.
  Fact 0.0: The Nassau Herald is a weekly newspaper.
  Fact 0.1: The Nassau Herald serves the Five Towns communities of Nassau County.
  Fact 0.2: The Five Towns communities of Nassau County are Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood, and Atlantic Beach.
Sentence 1: Lawrence is a village in Nassau County, New York in the United States.
  Fact 1.0: Lawrence is a village.
  Fact 1.1: Lawrence is in Nassau County, New York, United States.
Sentence 2:  As of the 2010 United States Census, the village population was 6,483.
  Fact 2.0: As of the 2010 United States Census, the population of the village of Lawrence was 6,483.

Retrieved items:
<item 1>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 1>

<item 2>
The Hudson Reporter
The Hudson Reporter is a newspaper chain based in Hudson County, New Jersey.

 It is the only weekly newspaper chain in Hudson County and one of only two newspaper companies in this busy metropolitan area.

 "The Hudson Reporter" publications focus on local politics and community news.

 In addition to articles written by the staff, the papers print readers' letters to the editor.

 The oldest newspaper in the chain is the "Hoboken Reporter", founded in 1983.
</item 2>

<item 3>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 3>

<item 4>
Lawrence, Nassau County, New York
Lawrence is a village in Nassau County, New York in the United States.

 As of the 2010 United States Census, the village population was 6,483.
</item 4>

<item 5>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 5>

<item 6>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 6>

<item 7>
Anton Media Group
Anton Media Group, formerly Anton Community Newspapers, and also known by its legal name Long Island Community Newspapers, Inc. is a print media company based
in Mineola, NY on Long Island in Nassau County and produces 17 weekly newspapers in Nassau County as well as "Long Island Weekly" and special sections such as
Healthy Living, Camp & School, and Dining Guides.

 Anton Community Newspapers is one of the largest privately owned newspaper companies in New York State.
</item 7>

<item 8>
Floral Park, New York
Floral Park is an incorporated village in Nassau County, New York, United States, on Long Island.

 The neighborhood of Floral Park in the New York City borough of Queens, is adjacent to the village.

 The village is at the western border of Nassau County, and is located mainly in the Town of Hempstead, while the section north of Jericho Turnpike is within
the Town of North Hempstead.

 The population as of the US Census of 2010 is 15,863.
</item 8>

<item 9>
Nassau Herald
The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic
Beach.

 It is part of the Long Island Herald newspaper chain, which includes The Jewish Star and The Oyster Bay Guardian is owned by Richner Communications, and covers
Nassau County, New York.
</item 9>

<item 10>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 10>
=== END ITEM id=4ccd80e88ae82448 ===

=== ITEM id=6185880da249a012 ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province, People's Republic of China.
Sentence 1: Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.
  Fact 1.0: Hailar District's Mongolian name is ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ, romanized as Qayilar toɣoriɣ.
  Fact 1.1: Hailar District's Mongolian name in Cyrillic script is Хайлаар тойрог.
  Fact 1.2: Hailar District's Cyrillic name is Хайлаар дүүрэг, romanized as Khailaar düüreg.
  Fact 1.3: Hailar District is an urban district.
  Fact 1.4: Hailar District serves as the seat of the prefecture-level city Hulunbuir.
  Fact 1.5: Hailar District is located in northeastern Inner Mongolia, China.

Retrieved items:
<item 1>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 1>

<item 2>
Old Barag Banner
Old Barag Banner (Mongolian: ᠬᠠᠭᠤᠴᠢᠨ ᠪᠠᠷᠭᠤ ᠬᠣᠰᠢᠭᠤ Қаучин Бару қосиу "Qaɣučin Barɣu qosiɣu"; ) is a banner of northeastern Inner Mongolia, People's Republic of
China.

 It is located 28 km from Hailar District, the administrative centre of Hulunbuir City, which administers this banner.
</item 2>

<item 3>
Dali–Ruili Railway
The Dali–Ruili Railway or Darui Railway (), is a single-track electrified railroad under construction in Yunnan Province of Southwest China.

 The line is slated to run 336.39 km from Dali in central Yunnan to Ruili in southwestern Yunnan on the border with Myanmar.

 The line traverses rugged terrain, and bridges and tunnels will account for 75% of the total track length, including a 36-km tunnel through the Gaoligong
Mountains.

 Construction began in May 2011 and is scheduled to take six years.

 Cities and towns along route include Dali, Yangbi Yi Autonomous County, Yongping County, Baoshan, Mang City (also known as Mangshi or Dehong) and Ruili.
</item 3>

<item 4>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 4>

<item 5>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 5>

<item 6>
Hulunbuir
Hulunbuir or Hulun Buir (Mongolian: ; , "Hūlúnbèi'ěr") is a region that is governed as a prefecture-level city in northeastern Inner Mongolia, in China.

 Its administrative center is located at Hailar District, its largest urban area.

 Major scenic features are the high steppes of the Hulun Buir grasslands, the Hulun and Buir lakes (the latter partially in Mongolia), and the Khingan range.

 Hulun Buir borders Russia to the north and west, Mongolia to the south and west, Heilongjiang province to the east and Hinggan League to the direct south.

 Hulunbuir is a linguistically diverse area: next to Mandarin Chinese, Mongolian dialects such as Khorchin and Buryat, the Mongolic language Dagur and some
Tungusic languages are spoken there.
</item 6>

<item 7>
Hailar District
Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as
the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.

 Hulunbuir, due to its massive size, is a city in administrative terms only, being mainly grassland and rural.

 Hailar, therefore, is a "de facto" city.

 Hailar can also refer to the urban area around the Hailar district, with Hulunbuir being the wider geographical region that contains the urban area.
</item 7>

<item 8>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 8>

<item 9>
Ruili
Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.

 It is a major border crossing between China and Myanmar, with the town of Muse located across the border.
</item 9>

<item 10>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 10>
=== END ITEM id=6185880da249a012 ===

=== ITEM id=8e9cf2520bd4b79b ===
Gold sentences and facts:
Sentence 0: Dana Robins Ivey (born August 12, 1941) is an American actress.
  Fact 0.0: Dana Robins Ivey was born on August 12, 1941.
  Fact 0.1: Dana Robins Ivey is an American actress.
Sentence 1: Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.
  Fact 1.0: Two Weeks Notice is a 2002 film.
  Fact 1.1: Two Weeks Notice is an Australian-American film.
  Fact 1.2: Two Weeks Notice is a romantic comedy film.
  Fact 1.3: Two Weeks Notice stars Hugh Grant.
  Fact 1.4: Two Weeks Notice stars Sandra Bullock.

Retrieved items:
<item 1>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 1>

<item 2>
Two Weeks Notice
Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.

 The film was written and directed by Marc Lawrence, who had previously scripted two other box-office hits for Bullock, "Miss Congeniality" and "Forces of
Nature".

 Although critical response was mixed, the film received a successful box office run, both in the United States and globally.
</item 2>

<item 3>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 3>

<item 4>
Josann McGibbon
Josann McGibbon is an American screenwriter working in partnership with Sara Parriott.

 The team's first major success as a screenwriter was the early Brad Pitt film, "The Favor".

 Their biggest hits since then include "Three Men and a Little Lady" and "Runaway Bride".

 In 2007, McGibbon and Parriott co-wrote and produced the hit Debra Messing miniseries, "The Starter Wife".

 "The Starter Wife" received 10 Emmy nominations in 2007, including for best screenwriting, and won one Emmy Award.

 It was also nominated for Golden Globe and Writers Guild awards, and was then produced as a series, also on USA Network.

 McGibbon and Parriott wrote and co-produced the Disney Channel movie, "Descendants" which was directed by Kenny Ortega and premiered in July, 2015.

 On February, 2013, it won the Writers Guild of America Award in television for Outstanding Children's Long Form.

 They then wrote and executive-produced "Descendants 2", also directed by Kenny Ortega, which was simulcast on the Disney Channel, ABC, and the other cable
channels owned by Disney-ABC in July, 2017.
</item 4>

<item 5>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 5>

<item 6>
Dana (surname)
Dana ( or ) as a surname may have several origins.

 In England, it came from "dann", the valley of a meadow, and it may mean the dweller of that valley.

 In Continental Europe, it probably came from "Dane" (or Danish, from Denmark).

 This surname is related to Danese in Italy and it can be found mostly in the Piedmont region.

 It may also be a modification of Huguenot French origin, probably a variant of "d'Aunay", of geographical origin.

 It may be also a Gaelic patronymic, since it is a common forename in Ireland.

 Dana is a relatively common surname in the US, ranking 7161 out of 88,799 in the 1990 U.S. Census.

 Dana is also the Persian word for wisdom.
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
Return to the Blue Lagoon
Return to the Blue Lagoon is a 1991 American romance and adventure film starring Milla Jovovich and Brian Krause, produced and directed by William A. Graham.

 The film is a sequel to "The Blue Lagoon" (1980).

 The screenplay by Leslie Stevens was based on the novel "The Garden of God" by Henry De Vere Stacpoole.

 The original music score was composed by Basil Poledouris.

 The film's closing theme song "A World of Our Own" is performed by Surface featuring Bernard Jackson.

 The music was written by Barry Mann, and the lyrics were written by Cynthia Weil.
</item 8>

<item 9>
Two Weeks (The Office)
"Two Weeks" is the twenty-first episode of the fifth season of the television series "The Office", and the 93rd overall episode of the series.

 It originally aired on NBC in the United States on March 26, 2009.

 In this episode, Michael, who has given his two weeks' notice to Dunder Mifflin, tries to convince others in the office to quit and join him in starting a new
paper company.

 Meanwhile, Pam spends her day trying to put together the new photocopier and becomes frustrated with her job.
</item 9>

<item 10>
Dana Ivey
Dana Robins Ivey (born August 12, 1941) is an American actress.

 She is a five-time Tony Award nominee for her work on Broadway, and won the 1997 Drama Desk Award for Outstanding Featured Actress in a Play for her work in
both "Sex and Longing" and "The Last Night of Ballyhoo".

 Her film appearances include "The Color Purple" (1985), "Dirty Rotten Scoundrels" (1988), "The Addams Family" (1991), "Two Weeks Notice" (2002), "Rush Hour 3"
(2007) and "The Help" (2011).
</item 10>
=== END ITEM id=8e9cf2520bd4b79b ===

=== ITEM id=c4be8e664cf36a45 ===
Gold sentences and facts:
Sentence 0: Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.
  Fact 0.0: Karen Joy Fowler was born on February 7, 1950.
  Fact 0.1: Karen Joy Fowler is American.
  Fact 0.2: Karen Joy Fowler is an author of science fiction.
  Fact 0.3: Karen Joy Fowler is an author of fantasy.
  Fact 0.4: Karen Joy Fowler is an author of literary fiction.
Sentence 1: Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.
  Fact 1.0: Charles Bruce Chatwin was born on 13 May 1940.
  Fact 1.1: Charles Bruce Chatwin died on 18 January 1989.
  Fact 1.2: Charles Bruce Chatwin was English.
  Fact 1.3: Charles Bruce Chatwin was a travel writer.
  Fact 1.4: Charles Bruce Chatwin was a novelist.
  Fact 1.5: Charles Bruce Chatwin was a journalist.

Retrieved items:
<item 1>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 1>

<item 2>
Always (short story)
"Always" is a science fiction short story written by Karen Joy Fowler.

 Originally published in the April/May 2007 issue of "Asimov's Science Fiction", it won the 2007 Nebula Award for Best Short Story.

 It is collected in Fowler's "What I Didn't See".
</item 2>

<item 3>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 3>

<item 4>
Karen Joy Fowler
Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.

 Her work often centers on the nineteenth century, the lives of women, and alienation.
</item 4>

<item 5>
The entries profile writers, scholars, artists, publishers, and advocates. Highlights include computer scientist Nuria Oliver’s data-science leadership and
human-behavior research; Helen Keller’s childhood story “The Frost King”; Booker Prize winner Keri Hulme; Alice Walker’s Pulitzer-winning *The Color Purple* and
anti-FGM work; and Lucille Watahomigie’s Hualapai language revitalization. Others include novelists Sarra Manning and Kim Edwards, poet Tara Betts
</item 5>

<item 6>
Kate Atkinson’s 2015 *A God in Ruins*, companion to *Life After Life*, follows Teddy Todd in WWII and won the Costa Novel Award. Other works include John
Howard’s Central/Eastern European fiction; Conrad’s *Heart of Darkness*, Marlow’s Congo voyage and Kurtz obsession; Keri Hulme’s Booker-winning *The Bone
People*; Karen Joy Fowler’s PEN/Faulkner-winning, Booker-shortlisted *We Are All Completely Beside Ours
</item 6>

<item 7>
Bruce Chatwin
Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.

 His first book, "In Patagonia" (1977), established Chatwin as a travel writer, although he considered himself instead a storyteller, interested in bringing to
light unusual tales.

 He won the James Tait Black Memorial Prize for his novel "On the Black Hill" (1982) and his novel "Utz" (1988) was shortlisted for the Man Booker Prize.

 In 2008 "The Times" named Chatwin #46 on their list of "50 Greatest British Writers Since 1945."
</item 7>

<item 8>
William Gibson
William Ford Gibson (born March 17, 1948) is an American-Canadian speculative fiction writer and essayist widely credited with pioneering the science fiction
subgenre known as cyberpunk.

 Beginning his writing career in the late 1970s, his early works were noir, near-future stories that explored the effects of technology, cybernetics, and
computer networks on humans—a "combination of lowlife and high tech"—and helped to create an iconography for the information age before the ubiquity of the
Internet in the 1990s.

 Gibson notably coined the term "cyberspace" in his short story "Burning Chrome" (1982) and later popularized the concept in his acclaimed debut novel
"Neuromancer" (1984).

 These early works have been credited with "renovating" science fiction literature after it had fallen largely into insignificance in the 1970s.
</item 8>

<item 9>
The Jane Austen Book Club
The Jane Austen Book Club is a 2004 novel by American author Karen Joy Fowler.

 The story, which takes place near Sacramento, California, centers on a book club consisting of five women and one man who meet once a month to discuss Jane
Austen's six novels.

 It was a critical success and became a national bestseller.
</item 9>

<item 10>
We Are All Completely Beside Ourselves
We Are All Completely Beside Ourselves is a 2013 novel by the American writer Karen Joy Fowler.

 The novel won the 2014 PEN/Faulkner Award for Fiction<ref name="The Guardian - 3 April 2014 - Karen Joy Fowler wins PEN/Faulkner award for fiction"> </ref> and
was also short-listed for the 2014 Man Booker Prize.
</item 10>
=== END ITEM id=c4be8e664cf36a45 ===

=== ITEM id=f937b74f203e4f52 ===
Gold sentences and facts:
Sentence 0: The Wire Fox Terrier is a breed of dog, one of many terrier breeds.
  Fact 0.0: The Wire Fox Terrier is a breed of dog.
  Fact 0.1: The Wire Fox Terrier is one of many terrier breeds.
Sentence 1: The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.
  Fact 1.0: The West Siberian Laika, also called WSL, is a breed of hunting dog.
  Fact 1.1: The West Siberian Laika, also called WSL, is a breed of spitz type.

Retrieved items:
<item 1>
Wire Fox Terrier
The Wire Fox Terrier is a breed of dog, one of many terrier breeds.

 It is a fox terrier, and although it bears a resemblance to the smooth fox terrier, they are believed to have been developed separately.
</item 1>

<item 2>
West Siberian Laika
The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.

 Russian publications indicate that the term West Siberian Laika loosely applied to hunting dogs originating with the Mansi and Khanty people in Ural and West
Siberia, but there were no standards or registrations of WSL as such until 1930.

 Then WWll disrupted it for a while, but "systematic breeding with registrations" resumed after the war ended, in 1946.

 This was the time the breed began taking modern shape.

 Before that hunters only knew of Mansi Laika and Khanty Laika.

 In early 1960 many hunters in Ural still preferred the term Mansi Laika, when speaking of West Siberian Laika.

 In Russian language, the term Laika originated from the word "layat" that means to bark.

 The word Laika simply means "barker".

 Any hunting Laika is a bark pointer (pointing at animal of interest by barking and staying with the animal ).

 It is a versatile dog depending on use and environment, but in certain parts of the country they have become more specialized.
</item 2>

<item 3>
Podenco Canario
Podenco Canario (In English: "Canary Islands Hound", "Canarian Warren Hound") is a breed of dog originally from the Canary Islands.

 The Podenco Canario is still used in the Canary Islands today in packs as a hunting dog, primarily used for rabbit hunting.

 The word "podenco" is Spanish for "hound".

 "Canario" means "Canarian" or "of the Canaries".
</item 3>

<item 4>
Fox Terrier
Fox Terriers are two different breeds of the terrier dog type: the Smooth Fox Terrier and the Wire Fox Terrier.

 Both of these breeds originated in the 19th century from a handful of dogs who are descended from earlier varieties of British terriers, and are related to
other modern white terrier breeds.

 In addition, a number of breeds have diverged from these two main types of fox terrier and have been recognised separately, including the Jack Russell Terrier,
Miniature Fox Terrier and Rat Terrier.

 The Wire and Smooth Fox Terriers share similar characteristics, the main differences being in the coat and markings.

 They have been successful in conformation shows, more prominently in America than their homeland.
</item 4>

<item 5>
Welsh Springer Spaniel
The Welsh Springer Spaniel is a breed of dog and a member of the spaniel family.

 Thought to be comparable to the old Land Spaniel, they are similar to the English Springer Spaniel and historically have been referred to as both the Welsh
Spaniel and the Welsh Cocker Spaniel.

 They were relatively unknown until a succession of victories in dog trials by the breed increased its popularity.

 Following recognition by The Kennel Club in 1902, the breed gained the modern name of Welsh Springer Spaniel.

 The breed's coat only comes in a single colour combination of white with red markings, usually in a piebald pattern.

 Loyal and affectionate, they can become very attached to family members and are wary of strangers.

 Health conditions are limited to those common among many breeds of dog, although they are affected more than average by hip dysplasia and some eye conditions.

 They are a working dog, bred for hunting, and while not as rare as some varieties of spaniel, they are rarer than the more widely known English Springer
Spaniel with which they are sometimes confused.
</item 5>

<item 6>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 6>

<item 7>
Signal Circuit of Halleston
Signal Circuit of Halleston was a Wire Fox Terrier and winner of the 1926 Westminster Kennel Club Dog Show.

 This was the 50th Westminster show, and the fourth Fox Terrier to win best in show.
</item 7>

<item 8>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 8>

<item 9>
Teddy Roosevelt Terrier
The Teddy Roosevelt Terrier is a small to medium-sized American hunting terrier.

 Lower-set with shorter legs, more muscular, and heavier bone density than its cousin the American Rat Terrier.

 There is much diversity in the history of the Teddy Roosevelt Terrier breed and it shares a common early history with the American Rat Terrier, Fox Paulistinha
and Tenterfield Terrier.

 It is said the Rat Terrier background stems from the terriers or other dogs that were brought over by early English and other working class immigrants.

 Since the breed was a farm, hunting and utility dog there was little to no planned breeding other than breeding dogs with agreeable traits to each other in
order to produce the desired work ethic in the dog.

 It is assumed that the Feist (dog), Bull Terrier, Smooth Fox Terrier, Manchester Terrier, Whippet, Italian Greyhound, the now extinct English White Terrier,
Turnspit dog and or Wry Legged Terrier all share in the Teddy Roosevelt Terrier's ancestry.

 These early Ratting Terriers were then most likely bred to the Beagle or Beagle cross bred dogs (for increased scenting ability) and other dogs.
</item 9>

<item 10>
Pomeranian (dog)
The Pomeranian (often known as a Pom or Pom Pom) is a breed of dog of the Spitz type that is named for the Pomerania region in Germany and Poland in Central
Europe.

 Classed as a toy dog breed because of its small size, the Pomeranian is descended from the larger Spitz type dogs, specifically the German Spitz.

 It has been determined by the Fédération Cynologique Internationale to be part of the German Spitz breed; and in many countries, they are known as the
Zwergspitz ("Dwarf-Spitz").
</item 10>
=== END ITEM id=f937b74f203e4f52 ===

=== ITEM id=d4aaeeb279b3b779 ===
Gold sentences and facts:
Sentence 0: Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space as part of the Soviet Union's Interkosmos programme.
  Fact 0.0: Sigmund Werner Paul Jähn was born on 13 February 1937.
  Fact 0.1: Sigmund Werner Paul Jähn is a German cosmonaut.
  Fact 0.2: Sigmund Werner Paul Jähn is a pilot.
  Fact 0.3: In 1978 Sigmund Werner Paul Jähn became the first East German to fly in space.
  Fact 0.4: In 1978 Sigmund Werner Paul Jähn became the first German native to fly in space.
  Fact 0.5: The 1978 flight by Sigmund Werner Paul Jähn was part of the Soviet Union's Interkosmos programme.
Sentence 1: Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.
  Fact 1.0: Arne Christer Fuglesang was born on March 18, 1957.
  Fact 1.1: Arne Christer Fuglesang was born in Stockholm.
  Fact 1.2: Arne Christer Fuglesang is a Swedish physicist.
  Fact 1.3: Arne Christer Fuglesang is an ESA astronaut.
Sentence 2:  He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
  Fact 2.0: Arne Christer Fuglesang was first launched aboard the STS-116 Space Shuttle mission.
  Fact 2.1: That launch was on December 10, 2006, at 01:47 GMT.
  Fact 2.2: Arne Christer Fuglesang became the first Swedish citizen in space.

Retrieved items:
<item 1>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 1>

<item 2>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 2>

<item 3>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 3>

<item 4>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 4>

<item 5>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 5>

<item 6>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 6>

<item 7>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 7>

<item 8>
Eberhard Köllner
Eberhard Köllner (born 29 September 1939 in Stassfurt, Germany) was selected for Soyuz 31 as the backup for Sigmund Jähn.
</item 8>

<item 9>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 9>

<item 10>
Breathing Your Love
"Breathing Your Love" is a song by the Swedish singer Darin featuring vocals by singer Kat DeLuna and the first single from "Flashback".

 The song co-written by Darin with RedOne, Bilal Hajji and Novel was released to radio stations and as a digital download in Sweden on 8 October 2008.

 The single was released in Finland and is also the first single from Darin to be released in the United Kingdom in January 2010.

 In October 2009 Swedish astronaut Christer Fuglesang took 2 copies of the single into space with him, as his daughter is a Darin fan, a photo of the disc can
be seen with earth in the distance can be found on the official Darin website.
</item 10>
=== END ITEM id=d4aaeeb279b3b779 ===

=== ITEM id=464d8c08d2848bce ===
Gold sentences and facts:
Sentence 0: Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also in Halland.
  Fact 0.0: Spettekaka is also called spettkaka.
  Fact 0.1: In native Scanian, spettekaka is called "spiddekaga".
  Fact 0.2: Spettekaka is a local dessert.
  Fact 0.3: Spettekaka is a local dessert of the southern parts of Sweden.
  Fact 0.4: Spettekaka is chiefly found in the province of Scania (Skåne).
  Fact 0.5: Spettekaka is also found in Halland.
Sentence 1: Scanian (   ) is a closely related group of South Swedish dialects spoken in the province of Scania in southern Sweden.
  Fact 1.0: Scanian is a closely related group of South Swedish dialects.
  Fact 1.1: Scanian dialects are spoken in the province of Scania.
  Fact 1.2: Scanian dialects are spoken in southern Sweden.
Sentence 2:  Scanian formed part of the old Scandinavian dialect continuum and are by most historical linguists considered to be an East Danish dialect group, but due to the modern-era influence from Standard Swedish in the region and because traditional dialectology in the Scandinavian countries normally has not considered isoglosses that cut across state borders, the Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.
  Fact 2.0: Scanian formed part of the old Scandinavian dialect continuum.
  Fact 2.1: Most historical linguists consider Scanian to be an East Danish dialect group.
  Fact 2.2: Standard Swedish has influenced the Scanian dialects in the region in the modern era.
  Fact 2.3: The Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.
  Fact 2.4: The Scanian dialects have been treated as South Swedish due to the modern-era influence from Standard Swedish in the region.
  Fact 2.5: The Scanian dialects have been treated as South Swedish because traditional dialectology in the Scandinavian countries normally has not considered isoglosses that cut across state borders.

Retrieved items:
<item 1>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 1>

<item 2>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 2>

<item 3>
Scanian dialect
Scanian (   ) is a closely related group of South Swedish dialects spoken in the province of Scania in southern Sweden.

 Scanian formed part of the old Scandinavian dialect continuum and are by most historical linguists considered to be an East Danish dialect group, but due to
the modern-era influence from Standard Swedish in the region and because traditional dialectology in the Scandinavian countries normally has not considered
isoglosses that cut across state borders, the Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.

 However, many of the early Scandinavian linguists, including Adolf Noreen and G. Sjöstedt, classified it as "South Scandinavian", and some linguists, such as
Elias Wessén, also considered Old Scanian a separate language, classified apart from both Old Danish and Old Swedish.
</item 3>

<item 4>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 4>

<item 5>
Hakata dialect
Hakata dialect (博多弁 , Hakata-ben ) is a Japanese dialect spoken in Fukuoka city.

 Hakata dialect originated in Hakata commercial district, while a related Fukuoka dialect (福岡弁 , Fukuoka-ben ) was spoken in the central district.

 Hakata dialect has spread throughout the city and its suburbs.

 Most of Japanese regard Hakata dialect as the dialect typical of Fukuoka, so it is sometimes called "Fukuoka-ben".
</item 5>

<item 6>
Skogsrå
The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical
creature of the forest, or Rå, in Swedish folklore.
</item 6>

<item 7>
Scoliopus
Scoliopus, or fetid adderstongue, is a genus of plant within the Liliaceae family consisting of two species, "Scoliopus bigelovii" and "S. hallii."

 Both are found in deep shaded forests, primarily in the coastal counties of the western United States from central California to northern Oregon.

 The name ""Scoliopus"" derives from the Greek words "skolios" and "pous", meaning curved foot, a reference to the shape of the pedicel.

 Taxonomists believe that "Scoliopus" is closely related to "Calochortus", "Prosartes", "Streptopus" and "Tricyrtis", which all have creeping rhizomes as well
as styles that divide at the tip.
</item 7>

<item 8>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 8>

<item 9>
Spettekaka
Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also
in Halland.

 It is an important part of the Scanian culinary heritage.

 The name means "cake on a spit", and this describes the method of preparation: it is the Swedish variation on the spit cake.
</item 9>

<item 10>
South Bergish
South Bergish (German: "Südbergische Dialekte ") or Upper Bergish (German: "Oberbergische Dialekte ") is a group of German dialects of the Bergisches Land
Region East of the Rhine and approximately south of the Wupper and north of the Sieg.

 These dialects are part of the Ripuarian group and thus are also called East Ripuarian.

 Ripuarian dialects are also spoken west of the Rhine up to the German border, and in some small areas next to the respective borders in Belgium and in the
Netherlands.

 Ripuarian Bergish dialects belong to the Middle German group, and thus are varieties of High German, where they belong to the northmost ones.

 In the North, they border to the East Bergish and the West Bergish aka Low Bergish language groups, which are part of the Low Franconian group like Dutch.
</item 10>
=== END ITEM id=464d8c08d2848bce ===
