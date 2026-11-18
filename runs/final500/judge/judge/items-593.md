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

This file has 10 items.

=== ITEM id=bbeeadb64f587668 ===
Gold sentences and facts:
Sentence 0: The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.
  Fact 0.0: The Shins are an American indie rock band.
  Fact 0.1: The Shins are from Albuquerque, New Mexico.
  Fact 0.2: The Shins were formed in 1996.
Sentence 1: Paramore is an American rock band from Franklin, Tennessee, formed in 2004.
  Fact 1.0: Paramore is an American rock band.
  Fact 1.1: Paramore is from Franklin, Tennessee.
  Fact 1.2: Paramore was formed in 2004.

Retrieved items:
<item 1>
Paramore, formed in Franklin, Tennessee, in 2004, centers on Hayley Williams, guitarist Taylor York, and drummer Zac Farro, who rejoined in 2017. The band has
five studio albums, from *All We Know Is Falling* (2005) to *After Laughter* (2017), whose bright pop sound contrasts with themes of anxiety and depression.
Notable songs include “The Only Exception,” “Grow Up,” and “Hate to See Your Heart
</item 1>

<item 2>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 2>

<item 3>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 3>

<item 4>
After Laughter
After Laughter is the fifth studio album by American rock band Paramore.

 It was released on May 12, 2017, through Fueled by Ramen as a follow-up to "Paramore", their 2013 self-titled album.

 The album was produced by guitarist Taylor York alongside previous collaborator, Justin Meldal-Johnsen.

 It is the band's first album since the return of drummer Zac Farro, who left the band with his brother Josh in 2010, and the departure of former bassist Jeremy
Davis, who left the band in 2015.

 "After Laughter" represents a complete departure from the usual pop punk and alternative rock sound of their previous releases.

 The album touches on themes of exhaustion, depression and anxiety, contrasting the upbeat and vibrant sound of the record.
</item 4>

<item 5>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 5>

<item 6>
Singles Club (EP)
Singles Club is an EP released by American rock band Paramore.

 The songs were released as promotional singles between October and December 2011, culminating in the release of a box set containing the three constituent
songs plus "Monster", which was recorded during the same sessions.

 The EP and box set were released on December 14, 2011 on Paramore's website.

 The EP is the first release of new material by Paramore not to include band members Josh and Zac Farro.
</item 6>

<item 7>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 7>

<item 8>
Paramore (album)
Paramore is the self-titled fourth studio album by American rock band Paramore.

 It was released on April 5, 2013, through Fueled by Ramen as a follow-up to "Brand New Eyes" (2009).

 It is their first full-length album released after the departure of co-founders Josh and Zac Farro in 2010.

 Recorded between April and November 1, 2012, the album was described by the band as being a "statement" and a reintroduction of the band to the world and to
themselves.

 It is the only album without Zac Farro on drums, and the final album recorded with bassist Jeremy Davis before he left the band in 2015.
</item 8>

<item 9>
List of songs recorded by Paramore
American rock band Paramore have recorded songs for five studio albums, a box set, an extended play and two soundtrack albums.

 In 2002, at age 13, vocalist Hayley Williams moved to Franklin, Tennessee, where she met brothers Josh Farro and Zac Farro.

 The band was officially formed by Josh Farro (lead guitar and backing vocals), Zac Farro (drums), Jeremy Davis (bass guitar) and Williams (lead vocals) in
2004, with the later addition of Williams' neighbor Jason Bynum (rhythm guitar).

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling" that year.

 Three singles were released to promote the album.
</item 9>

<item 10>
Paramore discography
The American rock band Paramore has released five studio albums, three extended plays, two live albums, eighteen singles, one video album, and nineteen music
videos.

 The band was formed in Franklin, Tennessee, in 2004 by lead vocalist Hayley Williams with guitarists Josh Farro and Taylor York, bassist Jeremy Davis, and
drummer Zac Farro.

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling".

 Three singles were released from the album, but none of them charted.

 The album did not chart in the "Billboard" 200 either, although it peaked at number thirty in the "Billboard" Top Heatseekers.

 "All We Know Is Falling" received Gold certification in the United Kingdom and in July 2014 the RIAA certified the album Gold in the United States.
</item 10>
=== END ITEM id=bbeeadb64f587668 ===

=== ITEM id=d58b3018e2e1bf5f ===
Gold sentences and facts:
Sentence 0:  It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
  Fact 0.0: The described species is common across temperate parts of Eurasia, northern Africa, and the Americas.
  Fact 0.1: The described species is common across subtropical parts of Eurasia, northern Africa, and the Americas.
  Fact 0.2: The described species is naturalized in Australia.
  Fact 0.3: The described species is naturalized on various islands.
Sentence 1: Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South America.
  Fact 1.0: Echeveria is a large genus of flowering plants.
  Fact 1.1: Echeveria is in the Crassulaceae family.
  Fact 1.2: Echeveria is native to semi-desert areas of Central America.
  Fact 1.3: Echeveria is native to semi-desert areas of Mexico.
  Fact 1.4: Echeveria is native to semi-desert areas of northwestern South America.

Retrieved items:
<item 1>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 1>

<item 2>
Agastache urticifolia
Agastache urticifolia is a species of flowering plant in the mint family known by the common name nettleleaf giant hyssop or horse mint.

 It is native to western North America from British Columbia to California to Colorado, where it grows in many habitat types.

 This is an aromatic perennial herb growing an erect stem with widely spaced leaves, each lance-shaped to nearly triangular and toothed.

 The leaves are up to 8 centimeters long and 7 wide.

 The inflorescence is a dense spike of many flowers.

 Each flower has long sepals tipped with bright purple and tubular corollas in shades of pink and purple.

 The fruit is a light brown, fuzzy nutlet about 2 millimeters long.

 The plant was used medicinally by several Native American groups, especially the leaves.
</item 2>

<item 3>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 3>

<item 4>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 4>

<item 5>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 5>

<item 6>
Echeveria setosa
Echeveria setosa, the Mexican fire cracker, is a species of flowering plant in the Crassulaceae family, native to semi-desert areas of Mexico and common
throughout Puebla.
</item 6>

<item 7>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 7>

<item 8>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 8>

<item 9>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 9>

<item 10>
Echeveria elegans
Echeveria elegans (Mexican snow ball, Mexican gem, white Mexican rose) is a species of flowering plant in the Crassulaceae family, native to semi-desert
habitats in Mexico.
</item 10>
=== END ITEM id=d58b3018e2e1bf5f ===

=== ITEM id=0aa523c5ac6d6390 ===
Gold sentences and facts:
Sentence 0: Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.
  Fact 0.0: Francis Keogh Gleason was born on April 14, 1906.
  Fact 0.1: Francis Keogh Gleason died on December 18, 1982.
  Fact 0.2: Francis Keogh Gleason was a resident set decorator at MGM studios.
  Fact 0.3: Francis Keogh Gleason was a resident set decorator at MGM studios for over 40 years.
Sentence 1:  In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi" in 1958) and was nominated an additional 3 times.
  Fact 1.0: Francis Keogh Gleason won 4 Academy Awards during his time at MGM.
  Fact 1.1: Francis Keogh Gleason won an Academy Award for 'An American in Paris' in 1951.
  Fact 1.2: Francis Keogh Gleason won an Academy Award for 'The Bad and the Beautiful' in 1952.
  Fact 1.3: Francis Keogh Gleason won an Academy Award for 'Somebody Up There Likes Me' in 1956.
  Fact 1.4: Francis Keogh Gleason won an Academy Award for 'Gigi' in 1958.
  Fact 1.5: Francis Keogh Gleason was nominated for an Academy Award an additional 3 times.
Sentence 2: Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.
  Fact 2.0: Somebody Up There Likes Me is a 1956 American drama film.
  Fact 2.1: Somebody Up There Likes Me is based on the life of middleweight boxing legend Rocky Graziano.

Retrieved items:
<item 1>
Rocky Graziano
Thomas Rocco Barbella (January 1, 1919 – May 22, 1990), better known as Rocky Graziano, was an American professional boxer who held the World Middleweight
title.

 Graziano is considered one of the greatest knockout artists in boxing history, often displaying the capacity to take his opponent out with a single punch.

 He was ranked 23rd on "The Ring" magazine list of the greatest punchers of all time.

 He fought many of the best middleweights of the era including Sugar Ray Robinson.

 His turbulent and violent life story was the basis of the 1956 Oscar-winning drama film, "Somebody Up There Likes Me", based on his 1955 autobiography of the
same title.
</item 1>

<item 2>
Rocky V
Rocky V is a 1990 American sports drama film.

 It is the fifth film in the "Rocky" series, written by and starring Sylvester Stallone, and co-starring Talia Shire, Stallone's real-life son Sage, and
real-life boxer Tommy Morrison, with Morrison in the role of Tommy Gunn, a talented yet raw boxer.

 Sage played Robert Balboa, whose relationship with his famous father is explored.

 After Stallone directed the second through fourth films in the series, "Rocky V" saw the return of John G. Avildsen, whose direction of "Rocky" won him an
Academy Award for Best Directing.
</item 2>

<item 3>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 3>

<item 4>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 4>

<item 5>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 5>

<item 6>
Somebody Up There Likes Me (1956 film)
Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.

 Joseph Ruttenberg was awarded a 1956 Oscar in the category of Best Cinematography (Black and White).

 The film also won the Oscar for Best Art Direction (Cedric Gibbons, Malcolm F. Brown, Edwin B. Willis, F. Keogh Gleason).

 It was directed by Robert Wise.
</item 6>

<item 7>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 7>

<item 8>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 8>

<item 9>
The Man with the Golden Arm
The Man with the Golden Arm is a 1955 American drama film with elements of film noir, based on the novel of the same name by Nelson Algren, which tells the
story of a drug addict who gets clean while in prison, but struggles to stay that way in the outside world.

 Although the addictive drug is never identified in the film, according to the American Film Institute "most contemporary and modern sources assume that it is
heroin", in contrast to Algren's book which named the drug as morphine.

 The film stars Frank Sinatra, Eleanor Parker, Kim Novak, Arnold Stang and Darren McGavin.

 It was adapted for the screen by Walter Newman, Lewis Meltzer and Ben Hecht (uncredited), and directed by Otto Preminger.

 The film's initial release sparked controversy due to its serious, in-depth treatment of the then-taboo subject of drug addiction.
</item 9>

<item 10>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 10>
=== END ITEM id=0aa523c5ac6d6390 ===

=== ITEM id=30cceb424799c083 ===
Gold sentences and facts:
Sentence 0:  It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that position for three weeks.
  Fact 0.0: The song was released as the lead single from the fourth studio album of its act, "Elephant."
  Fact 0.1: The song was released in March 2003.
  Fact 0.2: The song reached number one on the Modern Rock Tracks.
  Fact 0.3: The song maintained its number one position on the Modern Rock Tracks for three weeks.
Sentence 1:  Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album charts.
  Fact 1.0: The album "Elephant" was released on April 1, 2003.
  Fact 1.1: The album "Elephant" was released on V2 Records.
  Fact 1.2: The album "Elephant" garnered near unanimous critical acclaim.
  Fact 1.3: The album "Elephant" garnered commercial success.
  Fact 1.4: The album "Elephant" received a nomination for Album of the Year at the 46th Grammy Awards in 2004.
  Fact 1.5: The album "Elephant" won Best Alternative Music Album at the 46th Grammy Awards in 2004.
  Fact 1.6: The album "Elephant" peaked at No. 6 on the US "Billboard" charts.
  Fact 1.7: The album "Elephant" topped the UK album charts.

Retrieved items:
<item 1>
Elephant (album)
Elephant is the fourth album by the American alternative rock duo The White Stripes.

 Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the
Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album
charts.
</item 1>

<item 2>
Seven (Poco album)
Seven is the sixth studio album (seventh overall) by American country rock band Poco.

 It is the first album they made after leader Richie Furay left the band.

 The front cover was designed by Phil Hartman.
</item 2>

<item 3>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 3>

<item 4>
Seven Nation Army
"Seven Nation Army" (also stylized as "7 Nation Army") is a song by American rock duo The White Stripes.

 It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that
position for three weeks.

 It also became the third best-performing song of the decade on the same chart.

 It was well received commercially as well, and won the Grammy Award for Best Rock Song.
</item 4>

<item 5>
Grammy Award for Best Female Rap Solo Performance
The Grammy Award for Best Female Rap Solo Performance was an honor presented to female recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958 and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 5>

<item 6>
The Boys Are Back (The Oak Ridge Boys album)
The Boys Are Back is the thirtieth studio album of country music group The Oak Ridge Boys.

 It was released in 2009 under the Spring Hill Music Group label.

 The album marked the group's return to secular country music after releasing gospel albums since 1992.

 The track "Seven Nation Army," a cover of The White Stripes's 2003 song from their album "Elephant", was released as the album's first single.
</item 6>

<item 7>
Meg White
Megan Martha "Meg" White (born December 10, 1974) is an American drummer and occasional singer known for her work with Jack White in the Detroit rock duo The
White Stripes.

 On an impulse, she played on Jack's drums in 1997.

 The two decided to form a band and began performing two months later, calling themselves The White Stripes because of their last name and Meg's affinity for
peppermint candy.

 The band quickly became a Detroit underground favorite, before reaching national, then international fame.

 White has been nominated for various awards as a part of the White Stripes, and has received four Grammy Awards.
</item 7>

<item 8>
Premio Lo Nuestro 2003
Premio Lo Nuestro 2003 was the 15th anniversary of the awards.

 the show was hosted by Mexican presenters Marco Antonio Regil and Adal Ramones.

 Juanes, Thalía, Marc Anthony, Pilar Montenegro, Sin Bandera, Banda el Recodo and other Latin music greats gave electrifying performances.

 In the show, there was 36 awards winners with 135 nominations.

 In Pop genre, Awards was given for : Album of the Year, Best Male Artist, Best Female Artist, Best Group or duo, Best New Artist and Song of the Year.

 In Rock Genre : Best Rock Album and Best rock Performer of the Year.

 In Tropical genre : Best Tropical Album of the Year, Best Tropical Male Artist, Best Tropical Female Artist, Best Tropical Group or Duo of the Year, Best
Tropical New Artist, Tropical Song of the Year, Best Merengue Performance, Best Salsa Performance and Best Traditional Performance.

 Juanes was the biggest winner of night, took home four awards Best Pop Male Artist, Best Music Video, Best Rock Performance, and Pop Song of the Year .
</item 8>

<item 9>
Grammy Award for Best Male Rap Solo Performance
The Grammy Award for Best Male Rap Solo Performance was an honor presented to male recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958, and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 9>

<item 10>
Late Night Tales: The Flaming Lips
Late Night Tales: The Flaming Lips is a compilation album compiled by the members of The Flaming Lips, featuring songs by various artists.

 The album was released on March 7, 2005 and it features one new Flaming Lips recording, a cover of the White Stripes' "Seven Nation Army".
</item 10>
=== END ITEM id=30cceb424799c083 ===

=== ITEM id=4e59590bcc512cbe ===
Gold sentences and facts:
Sentence 0: Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.
  Fact 0.0: Corn Ranch is a spaceport.
  Fact 0.1: Corn Ranch is located in the West Texas town of Van Horn, Texas.
  Fact 0.2: Flight tests of the New Shepard are carried out at Corn Ranch.
  Fact 0.3: Flight tests of the New Shepard are carried out by Blue Origin.
Sentence 1: Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its headquarters in Kent, Washington.
  Fact 1.0: Blue Origin is an American privately funded aerospace manufacturer.
  Fact 1.1: Blue Origin is a spaceflight services company.
  Fact 1.2: Blue Origin was set up by Amazon.com founder Jeff Bezos.
  Fact 1.3: Blue Origin has its headquarters in Kent, Washington.

Retrieved items:
<item 1>
The entries describe U.S. and Saudi airports and authorities: Atlanta South (formerly Tara Field); remote Yakutat, served through Essential Air Service;
Wilmington, NC; Pennsylvania’s SARAA, overseeing four airports; Ta’if, a regional airport with international flights; Raleigh Exec, a reliever; Texarkana,
serving general aviation and American Eagle; Pensacola, despite its name, without direct international flights; Wilmington, Delaware; Columbus (formerly Port
Columbus); Amarillo, renamed for
</item 1>

<item 2>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 2>

<item 3>
Register Cliff, Wyoming, was an Oregon Trail landmark where emigrants carved names; roughly 500,000 used western trails from 1843–1869. Lafayette Regional
Airport serves Lafayette, Louisiana. Northern California TRACON manages regional air traffic, including major Bay Area and Sacramento airports. Blue Origin
tests New Shepard at Corn Ranch, Texas. Nevada’s Groom Lake is a dry lake and Area 51 runway site; nearby Rachel is known for UFO tourism, and Groom Range rises
to
</item 3>

<item 4>
M7 Aerospace
M7 Aerospace LP is an aerospace company with its headquarters on the property of San Antonio International Airport in Uptown San Antonio, Texas, United States.
</item 4>

<item 5>
Northern California TRACON
Northern California TRACON (NCT) (Terminal Radar Approach Control), or NorCal TRACON for short, is an air traffic control facility that provides safety alerts,
separation, and sequencing of air traffic arriving, departing, and transiting the airspace and airports in Northern California.

 Located in Rancho Cordova near Sacramento, NCT controls airspace over 19000 square miles, and serves Reno International Airport, Sacramento International
Airport, San Jose International Airport, Oakland International Airport, and San Francisco International Airport, plus 19 other smaller airports with air traffic
control towers.

 NCT is the 3rd busiest TRACON in America.

 NorCal TRACON is the step between local control (in an airport's control tower) and Air Route Traffic Control Center (ARTCC), in this case, Oakland Center
(ICAO code: ZOA).

 San Francisco International Airport is the 2nd largest airport in California and the largest airport serving Northern California.
</item 5>

<item 6>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 6>

<item 7>
Corn Ranch
Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.

 The 165,000-acre (670 km²) land parcel was purchased by Internet billionaire Jeff Bezos.

 The first flight test took place on November 13, 2006 with the goal of providing commercial tourist flights.
</item 7>

<item 8>
SyberJet Aircraft
SyberJet Aircraft is the manufacturer of the SJ30 light business jet.

 The company's headquarters is in Cedar City, Utah adjacent to the Cedar City Regional Airport with additional engineering offices and manufacturing, service,
repair and fatigue test facilities near and on the San Antonio International Airport in San Antonio, Texas.
</item 8>

<item 9>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 9>

<item 10>
H-T Ranch
H-T Ranch is a historic ranch complex 10 mi west of Amidon, North Dakota.

 The ranch originally consisted of ten buildings; however, only the ranch house and log barn survive.

 The ranch served as the headquarters of the Little Missouri Cattle Company, which was run by Arthur Clark Hidekoper.

 Hidekober established the ranch in the 1880s, and by the end of the decade, it had become the "most notable" ranch in the area.

 The ranch house, called Shackford, was built in 1896; its uncommon style resembles a bungalow but also borrows from other styles such as the Stick style.

 By 1906, the "Fargo Forum" described the ranch as "the biggest and most important [ranch] in the state" and "one of the largest horse raising outlets in the
world".

 In the same year, Hidekoper sold the 70,000 acre ranch; the sale was the largest land deal in North Dakota history.

 After the sale, a land company reduced the ranch to 5000 acre ; it was later used as a dude ranch in the 1920s.
</item 10>
=== END ITEM id=4e59590bcc512cbe ===

=== ITEM id=d75b41c6b47d926b ===
Gold sentences and facts:
Sentence 0: Brunswick Golden Isles Airport (IATA: BQK, ICAO: KBQK, FAA LID: BQK) , previously known as Glynco Jetport, is a county-owned public-use airport located five nautical miles (9 km) north of the central business district of Brunswick, a city in Glynn County, Georgia, United States.
  Fact 0.0: Brunswick Golden Isles Airport has IATA code BQK.
  Fact 0.1: Brunswick Golden Isles Airport has ICAO code KBQK.
  Fact 0.2: Brunswick Golden Isles Airport has FAA LID BQK.
  Fact 0.3: Brunswick Golden Isles Airport was previously known as Glynco Jetport.
  Fact 0.4: Brunswick Golden Isles Airport is a county-owned public-use airport.
  Fact 0.5: Brunswick Golden Isles Airport is located five nautical miles (9 km) north of the central business district of Brunswick.
  Fact 0.6: Brunswick is a city in Glynn County, Georgia, United States.
Sentence 1: Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana, United States.
  Fact 1.0: Lafayette Regional Airport has IATA code LFT.
  Fact 1.1: Lafayette Regional Airport has ICAO code KLFT.
  Fact 1.2: Lafayette Regional Airport has FAA LID LFT.
  Fact 1.3: Lafayette Regional Airport is a public airport.
  Fact 1.4: Lafayette Regional Airport is two miles (4 km) southeast of Lafayette.
  Fact 1.5: Lafayette Regional Airport is in Lafayette Parish, Louisiana, United States.

Retrieved items:
<item 1>
College of Coastal Georgia
The College of Coastal Georgia (also known as CCGA) is a public college located at historical Brunswick, Georgia, United States and the surrounding historical
Golden Isles.

 The College was established in 1961 and opened in 1964, making it one of Georgia's newest state colleges.

 The college transitioned from a two-year community college into a four-year comprehensive institution and conferred its first baccalaureate degrees on May 7,
2011.
</item 1>

<item 2>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 2>

<item 3>
Atlanta South Regional Airport
Henry County Airport (FAA LID: KHMP) , is a public-use county airport located three nautical miles (6 km) west of the central business district of Hampton, a
city in Henry County, Georgia, United States.

 It was known as Clayton County Airport – Tara Field, which was the name still used by the Federal Aviation Administration (FAA) and the Georgia Department of
Transportation (GDOT) until approval of the Atlanta South Regional Airport name.

 The airport was renamed Atlanta South Regional Airport, which was approved by the airport board of commissioners in December 2011, and was approved by the GDOT
and the FAA before it went into effect.
</item 3>

<item 4>
Golden Isles of Georgia
The Golden Isles of Georgia are a group of four barrier islands and the mainland port city of Brunswick on the 100-mile-long coast of the U.S. state of Georgia
on the Atlantic Ocean.

 They include St. Simons Island, Sea Island, Jekyll Island, Little St. Simons Island, and Historic Brunswick.
</item 4>

<item 5>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 5>

<item 6>
Jekyll Island Airport
Jekyll Island Airport is a small general aviation airport located in Jekyll Island, GA.

 It is around 2 miles away from central Jekyll Island, GA.

 Due to its small runway size and the nearby Brunswick Golden Isles Airport, no commercial airlines fly there, but scenic tours of the island fly from there.

 The airport has 3 based aircraft, and around 74 aircraft operations a day.
</item 6>

<item 7>
Colonel's Island Railroad
In the late 1960s, the State of Georgia started an improvement project for the port at Brunswick, Georgia.

 In this project, the Colonel's Island Railroad (reporting mark CISD) was created as a part of the Georgia Ports Authority.

 The CISD operated 33 miles of track in and around the Brunswick area.

 In 1998, the railroad sold most of its assets and the Golden Isles Terminal Railroad began operating on the line.

 The Ports Authority retained ownership of the track, but the CISD was disbanded.
</item 7>

<item 8>
Brunswick Old Town Historic District
Brunswick Old Town Historic District is a historic district in Brunswick, Georgia.

 It was added to the National Register of Historic Places on April 26, 1979 and includes an area bounded by 1st Street, Bay Street, New Bay Street, H Street,
and Cochran Street (4th Ave. and G St., according to one source).

 Brunswick is one of Georgia's two deep-water ports and is the mainland city associated with the Golden Isles of Georgia, at the junction of I-95 and US 82.
</item 8>

<item 9>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 9>

<item 10>
Brunswick Golden Isles Airport
Brunswick Golden Isles Airport (IATA: BQK, ICAO: KBQK, FAA LID: BQK) , previously known as Glynco Jetport, is a county-owned public-use airport located five
nautical miles (9 km) north of the central business district of Brunswick, a city in Glynn County, Georgia, United States.

 It is mostly used for general aviation, but is also served by one commercial airline.
</item 10>
=== END ITEM id=d75b41c6b47d926b ===

=== ITEM id=36610c48b5f42a72 ===
Gold sentences and facts:
Sentence 0:  In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.
  Fact 0.0: The Eleanor and John Y. Brown Jr. Foundation gave a grant in 1972.
  Fact 0.1: The 1972 grant from the Eleanor and John Y. Brown Jr. Foundation led to the construction of the current facility.
Sentence 1:  He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.
  Fact 1.0: The 55th governor of Kentucky served from 1979 to 1983.
  Fact 1.1: The 55th governor of Kentucky may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.

Retrieved items:
<item 1>
David Snowdon
David A. Snowdon (born 1952), is an epidemiologist and professor of neurology at the Sanders-Brown Center on Aging at the University of Kentucky.

 His research interests include antioxidants and aging, and the neuropathology of Alzheimer's disease, especially predictive factors in early life and the role
of brain infarction.
</item 1>

<item 2>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 2>

<item 3>
VIP's
VIP's, alternatively written Vip's, is a defunct restaurant chain in the Western United States that operated from 1968 until the late 1980s, based in Salem,
Oregon.

 With more than 50 locations, it was once the largest restaurant chain based in Oregon.

 It was a Denny's-style restaurant, a type that was commonly known at that time as a "coffee shop" but is now more commonly known as a casual dining restaurant.

 Most restaurants were located near freeways and were open 24 hours.

 At its peak, the chain had locations in five states: Oregon, Washington, Idaho, Nevada and northern California.
</item 3>

<item 4>
McAlister's Deli
McAlister's Deli is an American chain of fast casual restaurants founded in 1989 in Oxford, Mississippi by retired dentist Dr. Don Newcomb.

 There are currently over 400 locations in 26 states, ranging from Virginia in the East to Florida in the South to Arizona in West to Michigan in the North.

 The menu includes deli sandwiches, "Texas-size" spuds (baked potatoes), soups, salads, and desserts, as well as catering items such as sandwich trays and boxed
lunches.

 The chain is also known for its McAlister's Famous Sweet Tea, which is available by the glass or by the gallon.
</item 4>

<item 5>
Benjamin A. Rogge
Benjamin A. Rogge (June 18, 1920 – November 17, 1980) was an American economist, college administrator, and libertarian writer, speaker and foundation advisor.

 Rogge received an A.B. degree from Hastings College and an M.A. from the University of Nebraska Lincoln.

 Rogge received his PhD in economics from Northwestern.

 At Wabash College, Rogge taught in the summer Institute for Professional Development, in addition to his usual teaching in economics.

 Rogge co-authored an economics principles textbook with John Van Sickle.

 One strength of the text is the account that it gives of Joseph Schumpeter's process of creative destruction.

 Rogge helped organize a series of lectures by Milton Friedman at Wabash that were eventually developed into Friedman's "Capitalism and Freedom" book.

 Much later, Rogge participated in a brainstorming session for Friedman's Free to Choose television series.

 Liberty Fund was founded with money from Pierre Goodrich, who sought advice from Rogge during the Fund's early years.

 Rogge served for many years as a Liberty Fund trustee.
</item 5>

<item 6>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
</item 6>

<item 7>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 7>

<item 8>
Rubin Schron
Rubin Schron, who goes by Ruby, is a New York City real estate investor, landlord and the founder of Cammeby's International Group.

 He has a personal net worth of well over $10 billion, according to data company Real Capital Analytics.

 The portfolio of Cammeby's, which Schron founded in 1967, includes office buildings, market-rate and government-subsidized apartment complexes, nursing homes,
the 16-building complex in Sunset Park now known as Industry City, a stake in the bottom half of Woolworth Building and industrial properties scattered across
Long Island.

 In 2013, Schron made an unsolicited and unsuccessful offer to buy the Empire State Building for $2 billion, but he has not had problems closing many other
deals throughout his long career.

 In 2003, an investment group led by Schron paid $600 million for a portfolio of about 6,000 outer-borough apartments from Donald Trump.
</item 8>

<item 9>
Benjamin A. Rogge
Rogge served for many years as a Liberty Fund trustee.

 Thomas Sowell gives Rogge credit for encouraging him to write a book on economics and race.

 Rogge also was a frequent presenter at the seminars of the Foundation for Economic Education (FEE).

 FEE's founder, Leonard Read, thought of Rogge as Read's eventual successor, an outcome prevented by Read outliving Rogge.

 An extended (but somewhat grainy) video clip of a Rogge FEE lecture on "Competition and Monopoly" on YouTube illustrates the dry wit that made him a popular
speaker.

 Rogge attended 13 meetings of the influential international Mont Pelerin Society.

 Rogge helped produce, and narrated, a documentary on Adam Smith that was funded by Liberty Fund.

 Rogge wrote the introduction to a collection of quotations from Adam Smith.

 A collection of Rogge's speeches, often on topics in economics or education, was published under the title "Can Capitalism Survive?"

 Wabash College, where he taught for many years, established a speaker series in his honor.
</item 9>

<item 10>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 10>
=== END ITEM id=36610c48b5f42a72 ===

=== ITEM id=885aea60ecbbccae ===
Gold sentences and facts:
Sentence 0:  He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems."
  Fact 0.0: The recipient was awarded the 2012 Nobel Prize in Physics.
  Fact 0.1: The recipient shared the 2012 Nobel Prize in Physics jointly with Serge Haroche.
  Fact 0.2: The 2012 Nobel Prize in Physics was awarded for ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems.
Sentence 1:  Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.
  Fact 1.0: Since 2001, Serge Haroche has been a Professor at the Collège de France.
  Fact 1.1: Since 2001, Serge Haroche has held the Chair of Quantum Physics.

Retrieved items:
<item 1>
List of Nobel laureates in Chemistry
The Nobel Prize in Chemistry (Swedish: "Nobelpriset i kemi" ) is awarded annually by the Royal Swedish Academy of Sciences to scientists in the various fields
of chemistry.

 It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, who died in 1896.

 These prizes are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.

 As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Royal
Swedish Academy of Sciences.

 The first Nobel Prize in Chemistry was awarded in 1901 to Jacobus Henricus van 't Hoff, of the Netherlands.

 Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.

 In 1901, van 't Hoff received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 1>

<item 2>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 2>

<item 3>
Nobel Prize in Physics
The Nobel Prize in Physics (Swedish: "Nobelpriset i fysik" ) is a yearly award given by the Royal Swedish Academy of Sciences for those who conferred the most
outstanding contributions for mankind in the field of physics.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel
Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 3>

<item 4>
David J. Wineland
David Jeffrey Wineland (born February 24, 1944) is an American Nobel-laureate physicist at the National Institute of Standards and Technology (NIST) physics
laboratory.

 His work has included advances in optics, specifically laser cooling trapped ions and using ions for quantum computing operations.

 He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of
individual quantum systems."
</item 4>

<item 5>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 5>

<item 6>
Brian Josephson
Brian David Josephson, {'1': ", '2': ", '3': ", '4': "} (born 4 January 1940), is a Welsh theoretical physicist and professor emeritus of physics at the
University of Cambridge.

 Best known for his pioneering work on superconductivity and quantum tunnelling, he was awarded the Nobel Prize in Physics in 1973 for his prediction of the
Josephson effect, made in 1962 when he was a 22-year-old PhD student at Cambridge University.

 Josephson is the only Welshman to have won a Nobel Prize in Physics.

 He shared the prize with physicists Leo Esaki and Ivar Giaever, who jointly received half the award for their own work on quantum tunnelling.
</item 6>

<item 7>
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
</item 7>

<item 8>
List of Nobel laureates in Physiology or Medicine
Each recipient receives a medal, a diploma and a monetary award that has varied throughout the years.

 In 1901, von Behring received 150,782 SEK, which is equal to 7,731,004 SEK in December 2008.

 In 2013, the prize was awarded to James E. Rothman, Randy W. Schekman and Thomas C. Südhof; they were recognised "after discovering how cells precisely
transport material".

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 8>

<item 9>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 9>

<item 10>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 10>
=== END ITEM id=885aea60ecbbccae ===

=== ITEM id=7d6ae490d5bdffef ===
Gold sentences and facts:
Sentence 0: Pennsylvania Route 145 (PA 145) is a 20.89 mi long north–south state highway in the Lehigh Valley area of eastern Pennsylvania.
  Fact 0.0: Pennsylvania Route 145 (PA 145) is a state highway.
  Fact 0.1: Pennsylvania Route 145 is 20.89 miles long.
  Fact 0.2: Pennsylvania Route 145 runs north-south.
  Fact 0.3: Pennsylvania Route 145 is in the Lehigh Valley area of eastern Pennsylvania.
Sentence 1:  MacArthur Road is a divided highway; between U.S. Route 22 (US 22) and Eberhart Road, it is six lanes wide with a Jersey barrier and jughandles while the remainder of the road a four-lane divided highway.
  Fact 1.0: MacArthur Road is a divided highway.
  Fact 1.1: Between U.S. Route 22 (US 22) and Eberhart Road, MacArthur Road is six lanes wide.
  Fact 1.2: Between U.S. Route 22 and Eberhart Road, MacArthur Road has a Jersey barrier.
  Fact 1.3: Between U.S. Route 22 and Eberhart Road, MacArthur Road has jughandles.
  Fact 1.4: The remainder of MacArthur Road is a four-lane divided highway.
Sentence 2:  Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.
  Fact 2.0: Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.

Retrieved items:
<item 1>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 1>

<item 2>
Pennsylvania Canal (Delaware Division)
as part of the solution to the United States' first energy crisis.

 The Commonwealth of Pennsylvania built the Delaware canal to feed anthracite stone coal to energy-hungry Philadelphia as part of its transportation
infrastructure building plan known as the Main Line of Public Works—a legislative initiative creating a collection of self-reinforcing internal improvements to
commercial transportation capabilities.
</item 2>

<item 3>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 3>

<item 4>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 4>

<item 5>
Pennsylvania Route 145
Pennsylvania Route 145 (PA 145) is a 20.89 mi long north–south state highway in the Lehigh Valley area of eastern Pennsylvania.

 It connects Interstate 78 (I-78) and PA 309 in Lanark, Lehigh County to PA 248 in Weiders Crossing, Northampton County.

 PA 145 is the main north-south arterial into Allentown, the third-largest city in the state.

 The route enters the city on South 4th Street and follows multiple streets to downtown, where it follows the one-way pair of 6th Street northbound and 7th
Street southbound.

 North of Allentown in Whitehall Township, a seven-mile (11.2 km) portion of PA 145 is known as MacArthur Road, named in honor of General Douglas MacArthur.

 MacArthur Road is a divided highway; between U.S. Route 22 (US 22) and Eberhart Road, it is six lanes wide with a Jersey barrier and jughandles while the
remainder of the road a four-lane divided highway.

 MacArthur Road is the location of the main commercial center of the Lehigh Valley.
</item 5>

<item 6>
Duck Creek Aqueduct
Duck Creek Aqueduct, also known as the Metamora Aqueduct and Whitewater Canal Aqueduct, is a historic aqueduct carrying the Whitewater Canal over Duck Creek in
Metamora Township, Franklin County, Indiana.

 It is the only surviving covered wood aqueduct in the United States as such it is a form of covered bridge.

 It is a single-span Burr through truss aqueduct and measures approximately 90 feet long, 25 feet wide, and 25 feet deep.

 The original aqueduct was built between 1839 and 1843.

 The present structure was built in 1846, after the original aqueduct was washed out in a flood.

 It was strengthened in 1868, and repaired in 1901.

 After abandonment and deterioration, the Duck Creek Aqueduct was restored to its present appearance in 1946-1949.
</item 6>

<item 7>
Pennsylvania Route 145
MacArthur Road is the location of the main commercial center of the Lehigh Valley.

 North of Eagle Point, PA 145 becomes a two-lane undivided road that parallels the Lehigh River, crossing the river into Northampton County at Treichlers.

 The route continues along the east bank of the river and passes through Walnutport before reaching its northern terminus.

 PA 145 is dedicated as the Battle of the Bulge Veterans Memorial Highway in honor of the veterans who fought in the Battle of the Bulge.
</item 7>

<item 8>
Jersey barrier
A Jersey barrier or Jersey wall is a modular concrete or plastic barrier employed to separate lanes of traffic.

 It is designed to minimize vehicle damage in cases of incidental contact while still preventing the crossover case of a head-on collision.

 Jersey barriers are also used to reroute traffic and protect pedestrians and workers during highway construction, as well as temporary and semi-permanent
protections against landborne attack such as suicide vehicle bombs.

 A Jersey barrier is also known in the western United States as K-rail, or K-wall, a term borrowed from the California Department of Transportation
specification for temporary concrete traffic barriers, or colloquially as a Jersey bump.

 Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.
</item 8>

<item 9>
Susquehanna and Tidewater Canal
A copy of a detailed survey blueprint of the entire canal system including structures and property ownership details was donated (date unknown) by the Safe
Harbor Water Power Corporation to Millersville University (assessed on 10/06/2016 in the MU Archives at Sp.

 Coll.

 Map 386.409748 S128 Folio).

 The survey consists of 67 pages, 98 x 30 cm.

 and is undated but the assumption is the original was created while the canal was in use.
</item 9>

<item 10>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 10>
=== END ITEM id=7d6ae490d5bdffef ===

=== ITEM id=9a4f76eb5536f9f0 ===
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
Karen Joy Fowler
Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.

 Her work often centers on the nineteenth century, the lives of women, and alienation.
</item 1>

<item 2>
Lois McMaster Bujold
Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.

 She is one of the most acclaimed writers in her field, having won the Hugo Award for best novel four times, matching Robert A. Heinlein's record, not counting
his Retro Hugo.

 Her novella "The Mountains of Mourning" won both the Hugo Award and Nebula Award.

 In the fantasy genre, "The Curse of Chalion" won the Mythopoeic Award for Adult Literature and was nominated for the 2002 World Fantasy Award for best novel,
and both her fourth Hugo Award and second Nebula Award were for "Paladin of Souls".

 In 2011 she was awarded the Skylark Award.

 In 2013 she was awarded the Forry Award.

 In 2017 she won a Hugo Award for Best Series, for the Vorkosigan Saga.
</item 2>

<item 3>
Evonne Goolagong career statistics
This is a list of the main career statistics of Australian former tennis player Evonne Goolagong.

 During her career, which lasted from 1967 to 1983, Goolagong won seven singles titles at a Grand Slam event and was a runner-up on 11 occasions.

 In addition she won five Grand Slam doubles titles, partnering Margaret Court, Peggy Michel and Helen Gourlay, as well as one mixed doubles title with Kim
Warwick.

 In total she won 82 singles titles, 46 doubles titles and 4 mixed doubles titles.

 She achieved a No. 1 singles ranking for a two-week period in April–May 1976 although this was only officially recognized in 2007.

 She was a member of the Australian Federation Cup teams that won the cup in 1971, 1973 and 1974.
</item 3>

<item 4>
The Jane Austen Book Club
The Jane Austen Book Club is a 2004 novel by American author Karen Joy Fowler.

 The story, which takes place near Sacramento, California, centers on a book club consisting of five women and one man who meet once a month to discuss Jane
Austen's six novels.

 It was a critical success and became a national bestseller.
</item 4>

<item 5>
We Are All Completely Beside Ourselves
We Are All Completely Beside Ourselves is a 2013 novel by the American writer Karen Joy Fowler.

 The novel won the 2014 PEN/Faulkner Award for Fiction<ref name="The Guardian - 3 April 2014 - Karen Joy Fowler wins PEN/Faulkner award for fiction"> </ref> and
was also short-listed for the 2014 Man Booker Prize.
</item 5>

<item 6>
William Gibson
William Ford Gibson (born March 17, 1948) is an American-Canadian speculative fiction writer and essayist widely credited with pioneering the science fiction
subgenre known as cyberpunk.

 Beginning his writing career in the late 1970s, his early works were noir, near-future stories that explored the effects of technology, cybernetics, and
computer networks on humans—a "combination of lowlife and high tech"—and helped to create an iconography for the information age before the ubiquity of the
Internet in the 1990s.

 Gibson notably coined the term "cyberspace" in his short story "Burning Chrome" (1982) and later popularized the concept in his acclaimed debut novel
"Neuromancer" (1984).

 These early works have been credited with "renovating" science fiction literature after it had fallen largely into insignificance in the 1970s.
</item 6>

<item 7>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 7>

<item 8>
Always (short story)
"Always" is a science fiction short story written by Karen Joy Fowler.

 Originally published in the April/May 2007 issue of "Asimov's Science Fiction", it won the 2007 Nebula Award for Best Short Story.

 It is collected in Fowler's "What I Didn't See".
</item 8>

<item 9>
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
</item 9>

<item 10>
Bruce Chatwin
Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.

 His first book, "In Patagonia" (1977), established Chatwin as a travel writer, although he considered himself instead a storyteller, interested in bringing to
light unusual tales.

 He won the James Tait Black Memorial Prize for his novel "On the Black Hill" (1982) and his novel "Utz" (1988) was shortlisted for the Man Booker Prize.

 In 2008 "The Times" named Chatwin #46 on their list of "50 Greatest British Writers Since 1945."
</item 10>
=== END ITEM id=9a4f76eb5536f9f0 ===
