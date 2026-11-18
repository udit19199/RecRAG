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

=== ITEM id=eebe44f74ed9d779 ===
Gold sentences and facts:
Sentence 0: Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.
  Fact 0.0: Alice Malsenior Walker was born on February 9, 1944.
  Fact 0.1: Alice Malsenior Walker is an American novelist.
  Fact 0.2: Alice Malsenior Walker is an American short story writer.
  Fact 0.3: Alice Malsenior Walker is an American poet.
  Fact 0.4: Alice Malsenior Walker is an American activist.
Sentence 1: Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.
  Fact 1.0: Michael David Herr was born on April 13, 1940.
  Fact 1.1: Michael David Herr died on June 23, 2016.
  Fact 1.2: Michael David Herr was an American writer.
  Fact 1.3: Michael David Herr was an American war correspondent.
  Fact 1.4: Michael David Herr is known as the author of the memoir 'Dispatches', published in 1977.
  Fact 1.5: 'Dispatches' is a memoir of Michael David Herr's time as a correspondent for 'Esquire magazine' from 1967 to 1969.
  Fact 1.6: Michael David Herr was a correspondent during the Vietnam War.

Retrieved items:
<item 1>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 1>

<item 2>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 2>

<item 3>
Haniel Long
Haniel Clark Long (March 9, 1888 – October 17, 1956) was an American poet, novelist, publisher and academic.

 He is best known for his novella, "Interlinear to Cabeza de Vaca" (1936), a fictionalized account of the true story of a Spanish conquistador in 16th century
North America.
</item 3>

<item 4>
Tobe Levin
Alice Walker expressed appreciation for the text that shows worldwide solidarity with the novelist's literary abolition efforts in the early nineties.

 Levin has also teamed up with Maria Kiminta and photographer Britta Radike to publish a memoir and sourcebook, "Kiminta.

 A Maasai's Fight against Female Genital Mutilation."
</item 4>

<item 5>
Alice R. Ballard
Alice R. Ballard.

 (born June 16, 1945, Florence, South Carolina) is an American ceramicist based in Greenville, South Carolina.

 Much of her work is characterized by the organic earthenware forms of closed containers, pinch pots, platters, pods, teapots, totems, small work, vessels and a
series she refers to as her white work.

 Much of her work is finished with terra sigillata.

 Alice considers her art to be “a reflection of [her] relationship with natural forms.

 It is often the metamorphosis of Nature's forms, as they change from season to season, that attracts her to that universal world in which differing life forms
share similar qualities."

 Her aesthetic heavily references the mother/child/germination metaphor and also explores the more evocative realm of wonder and awe.
</item 5>

<item 6>
Alice Walker: Beauty in Truth
Alice Walker: Beauty in Truth is a documentary film directed by Pratibha Parmar, made by Kali Films production company.

 The film follows the life of the Pulitzer Prize-winning author, poet and activist Alice Walker.

 Shooting began in May 2011.

 It was aired on the BBC on Monday July 8, 2013, and on PBS on February 7, 2014.
</item 6>

<item 7>
Warrior Marks
Warrior Marks: Female Genital Mutilation and the Sexual Blinding of Women is a 1993 book by Alice Walker with Pratibha Parmar, who made an award-winning
documentary of the same name.

 Following on from her 1992 novel "Possessing the Secret of Joy", Walker undertakes a journey to parts of Africa where clitoridectomy is still practised.

 "Warrior Marks" is a harrowing work as Walker interviews women who have had the operation done and finally interviews a woman—circumcised herself—who performs
the operation.
</item 7>

<item 8>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
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
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 10>
=== END ITEM id=eebe44f74ed9d779 ===

=== ITEM id=94529644c2b3a718 ===
Gold sentences and facts:
Sentence 0: "Grow Some Funk of Your Own" is a song by Elton John.
  Fact 0.0: 'Grow Some Funk of Your Own' is a song by Elton John.
Sentence 1: Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.
  Fact 1.0: Sir Elton Hercules John was born Reginald Kenneth Dwight.
  Fact 1.1: Sir Elton Hercules John was born on 25 March 1947.
  Fact 1.2: Sir Elton Hercules John is an English singer.
  Fact 1.3: Sir Elton Hercules John is an English pianist.
  Fact 1.4: Sir Elton Hercules John is an English composer.
Sentence 2:  John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.
  Fact 2.0: Elton John owned Watford Football Club from 1976 to 1987.
  Fact 2.1: Elton John owned Watford Football Club from 1997 to 2002.

Retrieved items:
<item 1>
Afrodiziak
Afrodiziak was a British singing group composed of Caron Wheeler, Claudia Fontaine, and later Naomi Thompson, that was active in the 1980s.

 As a duo, Wheeler and Fontaine were best known for performing backing vocals on The Jam's final single "Beat Surrender" in 1982 (with whom they also performed
on their final tour) and Elvis Costello's 1983 album "Punch the Clock", especially its single, the international hit "Everyday I Write the Book".

 After the addition of a third member, Naomi Thompson, they went on to sing for acts such as Heaven 17 and Howard Jones.

 Afrodiziak performed backing vocals on the hit single Free Nelson Mandela, including the a cappella intro, staging it at The Tube (Channel4) in March 1984.

 Heaven 17's 1984 album, "How Men Are", featured them prominently, especially on the singles "Sunset Now" and "And That's No Lie".
</item 1>

<item 2>
Luton Airport (song)
Let's put the record straight on "Luton Airport" it was born out of an advert for Campari featuring Loraine Chase and other 'Celebrity' Actors.

 It was written off the back of the advert cashing in on its success .

 It was produced by Dave Dee at WEA records.

 The lead singer was not Dina Payne but a top TV dancer named Bee Rowley who worked with many of the leading TV choreographers of the 80's including Geoff
Richer, Nigel Lithco and Arlene Phillips- Dina along with two other girls in the group supplied the backing vocals.

 Cats UK with Bee Rowley as lead singer made a follow up cover version of " Sixteen Looking for Love" which failed to reach the charts.

 The group was disbanded.

 Despite their appearances on "T.O.P and making revenue from sales none of the 'Group received a penny for their hard work and time fronting and performing the
record because as often the song writers (NOT Dave Dee - the producer) took their royalties but didn't the artists.
</item 2>

<item 3>
Grow Some Funk of Your Own
"Grow Some Funk of Your Own" is a song by Elton John.

 It was released as a single in 1976 and comes from his album "Rock of the Westies".

 It shared its A-side status with "I Feel Like a Bullet (In the Gun of Robert Ford)".

 The song is considered to be one of John's heavier rock songs.

 The song went to #14 on the "Billboard" Hot 100, but in Britain broke a five-year run of successful singles by failing to reach the Top 50 despite extensive
radio play.

 Although the singles chart listed only "Funk" and not "Bullet", the latter song was given equal billing in later "Billboard" singles chart books.
</item 3>

<item 4>
Martin Degville
Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit
single in 1986 with "Love Missile F1-11" – and six other EMI single releases.

 Sputnik was formed with ex-Generation X bassist Tony James.
</item 4>

<item 5>
The Flowers of Romance (band)
The Flowers of Romance were an early punk band, formed in mid-1976 by Jo Faull and Sarah Hall.

 The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that
later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and
Viv Albertine, who went on to play in The Slits.

 Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
</item 5>

<item 6>
Blitz Kids
The Blitz Kids were a group of young people who frequented the weekly Blitz club-night in Covent Garden, London in 1979-80, and are credited with launching the
New Romantic subcultural movement.

 It was Andrew Czezowski (Ex manager of The Damned) and Susan Carrington who went on to start The Roxy club with Barry Jones that introduced Steve Strange and
Rusty Egan to the Blitz in Covent Garden, which started the New Romantic Movement.

 Among the many attendees were Rusty Egan, Steve Strange, Boy George, Marilyn, Alice Temple, Perri Lister, Princess Julia, Philip Sallon, Carl Teper and Martin
Degville (later to be the frontman of Tony James' Sigue Sigue Sputnik).

 Crucially, the club lay between two art colleges (St Martin's School and Central School) and became a testbed for student fashion designers who set London
ablaze during the 1980s.
</item 6>

<item 7>
The Happenings
The Happenings are a pop music group that originated in the 1960s.

 Members of the original group, created in the spring of 1961 and initially called "The Four Grads" because all had just graduated from high school in Paterson,
New Jersey, were Bob Miranda, David Libert, Tom Giuliano, and Ralph DiVito.

 In 1968 DiVito was replaced by Bernie LaPorta and Lenny Conforti also joined to play drums in the touring band.

 Both LaPorta and Conforti took a hiatus from the northern New Jersey band, The Emerald Experience, to play and tour with The Happenings.

 That lineup performed mostly at colleges and universities until 1970, when Libert left the band to manage other groups, including George Clinton and
Parliament-Funkadelic, Living Colour, Brian Auger, Vanilla Fudge, The Runaways (Cherie Currie, Joan Jett, Lita Ford), Mother's Finest, Alice Cooper and Evelyn
"Champagne" King.
</item 7>

<item 8>
Motown
Motown is an American record company.

 The record company was founded by Berry Gordy Jr. as Tamla Records on January 12, 1959, and was incorporated as Motown Record Corporation on April 14, 1960, in
Detroit, Michigan.

 The name, a portmanteau of "motor" and "town", has also become a nickname for Detroit.

 Motown played an important role in the racial integration of popular music as an African American-owned record label that achieved significant crossover
success.

 In the 1960s, Motown and its subsidiary labels (including Tamla Motown, the brand used outside the US) were the most successful proponents of what came to be
known as the Motown Sound, a style of soul music with a distinct pop influence.

 During the 1960s, Motown achieved spectacular success for a small record company: 79 records in the Top Ten of the "Billboard" Hot 100 record chart between
1960 and 1969.
</item 8>

<item 9>
Elton John
He has also composed music, produced records, and has occasionally acted in films.

 John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.

 He is an honorary Life President of the club, and in 2014 had a stand named after him at the club's home stadium.
</item 9>

<item 10>
South Liverpool F.C.
South Liverpool Football Club is a football club based in Liverpool, England, founded as a phoenix club of a club of the same name.

 They are currently members of the West Cheshire League Division One and play at Jericho Lane in Otterspool.

 The club's colours are white shirts, black shorts and red socks.
</item 10>
=== END ITEM id=94529644c2b3a718 ===

=== ITEM id=7f817dc2bb598163 ===
Gold sentences and facts:
Sentence 0: The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.
  Fact 0.0: The Shame of a City is a 2006 feature-length documentary.
  Fact 0.1: The Shame of a City premiered at the Philadelphia Film Festival.
Sentence 1: Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and preserved in the WPA Slave Narrative Collection.
  Fact 1.0: Unchained Memories is a 2003 documentary film.
  Fact 1.1: Unchained Memories is about the stories of former slaves.
  Fact 1.2: The former slaves were interviewed during the 1930s.
  Fact 1.3: The interviews were part of the Federal Writers' Project.
  Fact 1.4: The stories are preserved in the WPA Slave Narrative Collection.

Retrieved items:
<item 1>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 1>

<item 2>
Unchained Memories
Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and
preserved in the WPA Slave Narrative Collection.

 This HBO film interpretation directed by Ed Bell and Thomas Lennon is a compilation of slave narratives, narrated by actors, emulating the original
conversation with the interviewer.

 The slave narratives may be the most accurate in terms of the everyday activities of the enslaved, serving as personal memoirs of more than two thousand former
slaves.

 The documentary depicts the emotions of the slaves and what they endured.

 The "Master" had the opportunity to sell, trade, or kill the enslaved, for retribution should one slave not obey.
</item 2>

<item 3>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 3>

<item 4>
Graph query row:
work: Unchained Memories
year: 2003
</item 4>

<item 5>
Graph query row:
work: The Shame of a City
year: 2006
</item 5>

<item 6>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 6>

<item 7>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 7>

<item 8>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 8>

<item 9>
Romance of Celluloid
The Romance of Celluloid is a 1937 short black and white documentary film, narrated by Frank Whitbeck, which goes behind the scenes to look at the manufacture
of film and the making of motion pictures.

 The film was the first of the studio's "Romance of Celluloid" series, which also included;
</item 9>

<item 10>
The Shame of a City
The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.

 Filmmaker Tigre Hill chronicles the 2003 Philadelphia mayoral race between Democrat incumbent mayor John Street and Republican challenger Sam Katz.

 Early polls showed Katz with a small lead in this predominantly Democratic city but twenty-seven days before the election, an FBI bug was found in the mayor’s
office.

 The discovery at first seemed like a death knell to the Street campaign and a near certain victory for Katz.

 Yet this prediction was proven wrong when Street and his supporters successfully polarized the campaign by leveling accusations of instituational racial
prejudice and playing on historical skepticism of the Republican-controlled federal government.

 As a result, Street won re-election by a sixteen-point margin.
</item 10>
=== END ITEM id=7f817dc2bb598163 ===

=== ITEM id=a49bec65987289fe ===
Gold sentences and facts:
Sentence 0: The Wire Fox Terrier is a breed of dog, one of many terrier breeds.
  Fact 0.0: The Wire Fox Terrier is a breed of dog.
  Fact 0.1: The Wire Fox Terrier is one of many terrier breeds.
Sentence 1: The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.
  Fact 1.0: The West Siberian Laika, also called WSL, is a breed of hunting dog.
  Fact 1.1: The West Siberian Laika, also called WSL, is a breed of spitz type.

Retrieved items:
<item 1>
Fox Terrier
Fox Terriers are two different breeds of the terrier dog type: the Smooth Fox Terrier and the Wire Fox Terrier.

 Both of these breeds originated in the 19th century from a handful of dogs who are descended from earlier varieties of British terriers, and are related to
other modern white terrier breeds.

 In addition, a number of breeds have diverged from these two main types of fox terrier and have been recognised separately, including the Jack Russell Terrier,
Miniature Fox Terrier and Rat Terrier.

 The Wire and Smooth Fox Terriers share similar characteristics, the main differences being in the coat and markings.

 They have been successful in conformation shows, more prominently in America than their homeland.
</item 1>

<item 2>
Pomeranian (dog)
The Pomeranian (often known as a Pom or Pom Pom) is a breed of dog of the Spitz type that is named for the Pomerania region in Germany and Poland in Central
Europe.

 Classed as a toy dog breed because of its small size, the Pomeranian is descended from the larger Spitz type dogs, specifically the German Spitz.

 It has been determined by the Fédération Cynologique Internationale to be part of the German Spitz breed; and in many countries, they are known as the
Zwergspitz ("Dwarf-Spitz").
</item 2>

<item 3>
Wire Fox Terrier
The Wire Fox Terrier is a breed of dog, one of many terrier breeds.

 It is a fox terrier, and although it bears a resemblance to the smooth fox terrier, they are believed to have been developed separately.
</item 3>

<item 4>
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
</item 4>

<item 5>
Signal Circuit of Halleston
Signal Circuit of Halleston was a Wire Fox Terrier and winner of the 1926 Westminster Kennel Club Dog Show.

 This was the 50th Westminster show, and the fourth Fox Terrier to win best in show.
</item 5>

<item 6>
Old Jock
Old Jock (1859–1871), was a Fox Terrier famous during the late 19th and early 20th centuries.

 A mostly white dog, he ran briefly with a hunting kennel before becoming a show dog, most notably with a victory at the show which popularised the Fox Terrier.

 His main show rivalry was with a dog named Tartar, and along with a dog named Trap, the three were popular sires of the Fox Terrier breed.

 He was also involved in the early formation of the Jack Russell Terrier and the Dandie Dinmont Terrier breeds.
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 8>

<item 9>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 9>

<item 10>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 10>
=== END ITEM id=a49bec65987289fe ===

=== ITEM id=773456d5be09d593 ===
Gold sentences and facts:
Sentence 0: Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
  Fact 0.0: Ex parte Young is a United States Supreme Court case.
  Fact 0.1: Ex parte Young is cited as 209 U.S. 123.
  Fact 0.2: Ex parte Young is dated 1908.
  Fact 0.3: Ex parte Young allows suits in federal courts against officials acting on behalf of states of the union.
  Fact 0.4: Ex parte Young allows such suits to proceed despite the state's sovereign immunity.
  Fact 0.5: Ex parte Young allows such suits when the state acted unconstitutionally.
Sentence 1:  The case was a test of the separation of church and state with respect to education.
  Fact 1.0: The Ex parte Young case was a test of the separation of church and state.
  Fact 1.1: The test of church and state separation in the Ex parte Young case concerned education.

Retrieved items:
<item 1>
Ex parte Young
Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of
the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
</item 1>

<item 2>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 2>

<item 3>
Criminal law in the Chase Court
The Chase Court (1864–1873) issued thirty-five opinions in criminal cases over nine years, at a significantly higher rate than the Marshall Court or Taney Court
before it.

 Notable such cases include "Ex parte Milligan" (1866), "Pervear v. Massachusetts" (1866), "Ex parte McCardle" (1867, 1869), "Ex parte Yerger" (1868), and
"United States v. Kirby" (1868).
</item 3>

<item 4>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 4>

<item 5>
Graph query row:
case1: Ex parte Young
case1_relationships: []
case2: McCollum v. Board of Education
case2_relationships: []
shared_connections: []
direct_relationships: []
</item 5>

<item 6>
Criminal law in the Waite Court
Notable cases include "United States v. Cruikshank" (1875), "United States v. Reese" (1875), "Reynolds v. United States" (1878), "Wilkerson v. Utah" (1879), the
"Trade-Mark Cases" (1879), "Strauder v. West Virginia" (1880), "Pace v. Alabama" (1883), "United States v. Harris" (1883), "Ex parte Crow Dog" (1883), "Hurtado
v. California" (1884), "Clawson v. United States" (1885), "Yick Wo v. Hopkins" (1886), "United States v. Kagama" (1886), "Ker v. Illinois" (1886), and "Mugler
v. Kansas" (1887).
</item 6>

<item 7>
Seminole Tribe of Florida v. Florida
Seminole Tribe of Florida v. Florida, 517 U.S. 44 (1996) , was a United States Supreme Court case which held that Article One of the U.S. Constitution did not
give the United States Congress the power to abrogate the sovereign immunity of the states that is further protected under the Eleventh Amendment.

 Such abrogation is permitted where it is necessary to enforce the rights of citizens guaranteed under the Fourteenth Amendment as per "Fitzpatrick v. Bitzer"
427 U.S. 445 (1976).

 The case also held that the doctrine of "Ex parte Young", 209 U.S. 123 (1908), which allows state officials to be sued in their official capacity for
prospective injunctive relief, was inapplicable under these circumstances, because any remedy was limited to the one that Congress had provided.
</item 7>

<item 8>
Idaho v. Coeur d'Alene Tribe of Idaho
Idaho v. Coeur d'Alene Tribe of Idaho, 521 U.S. 261 (1997), was a United States Supreme Court case in which the Court held that the Coeur d'Alene Tribe could
not maintain an action against the state of Idaho to press its claim to Lake Coeur d'Alene due to the state's Eleventh Amendment immunity from suit,
notwithstanding the exception recognized in "Ex parte Young".

 The case was an important precedent for aboriginal title in the United States and sovereign immunity in the United States.
</item 8>

<item 9>
Chy Lung v. Freeman
Chy Lung v. Freeman, 92 U.S. 275 (1876) was a United States Supreme Court case where the Supreme Court ruled that the power to set rules surrounding
immigration, and to manage foreign relations, rested with the United States Federal Government, rather than with the states.

 The case has been cited in other Supreme Court cases related to government authority on matters relating to immigration policy and immigration enforcement,
most recently in "Arizona v. United States" (2012).
</item 9>

<item 10>
Duncan v. Kahanamoku
Duncan v. Kahanamoku, 327 U.S. 304 (1946), was a decision by the United States Supreme Court.

 It is often associated with the Japanese exclusion cases ("Hirabayashi v. United States", "Korematsu v. United States" and "Ex parte Endo") because it involved
wartime curtailment of fundamental civil liberties under the aegis of military authority, though in this case neither the plaintiff nor the nominal defendant
were Japanese.
</item 10>
=== END ITEM id=773456d5be09d593 ===

=== ITEM id=a78a8f08457c656a ===
Gold sentences and facts:
Sentence 0: The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR: "Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet Union) and South Korea (with the principal support of the United States).
  Fact 0.0: The Korean War is called 한국전쟁 in South Korean Hangul.
  Fact 0.1: The Korean War is called 韓國戰爭 in South Korean Hanja.
  Fact 0.2: The Korean War is romanized as 'Hanguk Jeonjaeng' (RR), meaning 'Korean War'.
  Fact 0.3: The Korean War is called 조국해방전쟁 in North Korean Chosŏn'gŭl.
  Fact 0.4: The Korean War is called 祖國解放戰爭 in North Korean Hancha.
  Fact 0.5: The Korean War is romanized as 'Choguk haebang chǒnjaeng' (MR), meaning 'Fatherland Liberation War'.
  Fact 0.6: The Korean War lasted from 25 June 1950 to 27 July 1953.
  Fact 0.7: The Korean War was a war between North Korea and South Korea.
  Fact 0.8: North Korea fought with the support of China and the Soviet Union.
  Fact 0.9: South Korea fought with the principal support of the United States.
Sentence 1: The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.
  Fact 1.0: The Western Allied invasion of Germany was coordinated by the Western Allies.
  Fact 1.1: The Western Allied invasion of Germany took place during the final months of hostilities in the European theatre of World War II.
Sentence 2:  The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.
  Fact 2.0: The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945.
  Fact 2.1: After crossing the Rhine, the Western Allies fanned out and overran all of western Germany.
  Fact 2.2: The Western Allies overran western Germany from the Baltic in the north to Austria in the south.
  Fact 2.3: The Germans surrendered on 8 May 1945.

Retrieved items:
<item 1>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 1>

<item 2>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 2>

<item 3>
South German Offensive
The South German Offensive is the general name of one of the final offensives of World War II in Europe.

 The offensive was led by the Seventh and Third armies of the United States along with the First Army of France.

 Soviet troops linked up with American forces in Czechoslovakia notably in the Battle of Slivice.

 The offensive was made by the US 6th Army Group to protect the US 12th Army Group's right flank and to prevent a German last stand in the Alps.

 However German resistance was much more fierce than in the north, which slowed the 6th Army Group's progress.

 However, by the end of April, many German divisions surrendered without a fight to the advancing American forces to avoid the inevitable destruction.

 The VI Corps of the Seventh Army linked up with the US Fifth Army, which fought through Italy, in the Alps as the Third Army advanced into Austria and
Czechoslovakia, where it linked up with Soviet forces advancing from the east.

 Fighting continued a few days after the Surrender of Germany on 8 May, due to German forces fighting west to surrender to the Americans instead of the Soviets.
</item 3>

<item 4>
Flight and expulsion of Germans (1944–50)
During the later stages of World War II and the post-war period, German citizens and people of German ancestry were expelled from various Eastern European
countries and sent to the remaining territory of Germany and Austria.

 After 1950, some emigrated to the United States, Australia, and other countries from there.

 The areas affected included the former eastern territories of Germany, which were annexed by Poland and the Soviet Union after the war, as well as Germans who
were living within the prewar borders of Poland, Czechoslovakia, Hungary, Romania, Yugoslavia, and the Baltic States.

 The Nazis had made plans—only partially completed before the Nazi defeat—to remove many Slavic and Jewish people from Eastern Europe and settle the area with
Germans.

 The post war expulsion of the Germans formed a major part of the geopolitical and ethnic reconfiguration of Eastern Europe in the aftermath of World War II,
that attempted to create ethnically homogeneous nations within redefined borders.

 Between 1944 and 1948 about 31 million people, including ethnic Germans ('Volksdeutsche') as well as German citizens ('Reichsdeutsche'), were permanently or
temporarily moved from Central and Eastern Europe.
</item 4>

<item 5>
21st Army Group
The 21st Army Group was a World War II British headquarters formation, in command of two field armies and other supporting units, consisting primarily of the
British Second Army and the First Canadian Army.

 Established in London during July 1943, under the command of Supreme Headquarters Allied Expeditionary Force (SHAEF), it was assigned to Operation Overlord,
the Western Allied invasion of Europe, and was an important Allied force in the European Theatre.

 The 21st Army Group operated in Northern France, Luxembourg, Belgium, the Netherlands and Germany from June 1944 until the end of the war in Europe in May
1945, after which it was redesignated the British Army of the Rhine (BAOR).
</item 5>

<item 6>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 6>

<item 7>
Western Allied invasion of Germany
The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.

 The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany
from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.

 This is known as the "Central Europe Campaign" in United States military histories.
</item 7>

<item 8>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 8>

<item 9>
Allied invasion of Italy
The Allied invasion of Italy was the Allied amphibious landing on mainland Italy that took place on 3 September 1943 during the early stages of the Italian
Campaign of World War II.

 The operation was undertaken by General Sir Harold Alexander's 15th Army Group (comprising General Mark W. Clark and General Bernard Montgomery's British
Eighth Army) and followed the successful invasion of Sicily.

 The main invasion force landed around Salerno on 9 September on the western coast in Operation Avalanche, while two supporting operations took place in
Calabria (Operation Baytown) and Taranto (Operation Slapstick).
</item 9>

<item 10>
Wayne W. Lambert
Wayne W. Lambert (born 1936) was an Air force Brigadier General (United States).

 He attended the United States Military Academy at West Point, New York, after already having served a year in the US Army.

 Lambert graduated from the Academy in 1959 and was commissioned a 2nd Lieutenant in the USAF that same year.

 He trained as a bomber pilot, receiving his Aviator badge in 1960.

 He has logged over 5,800 flying hours with the US Air Force.

 A seasoned combat veteran, Lambert flew over 225 combat missions in South East Asia as a B-52 crew member.

 He was awarded the Distinguished Flying Cross (United States) and received 10 awards of the Air Medal for his service during the Vietnamese War.

 After serving in a succession of lesser command positions, Lambert was promoted to the rank of Brigadier General in 1983, assuming command of the former
Strategic Air Command’s 7th Air Division, headquartered then at Ramstein Air Base in West Germany, and overseeing SAC operations throughout the European
theater.

 The Brigadier General retired from the US Air Force in 1989.
</item 10>
=== END ITEM id=a78a8f08457c656a ===

=== ITEM id=5eb8156b9d7faec3 ===
Gold sentences and facts:
Sentence 0: Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.
  Fact 0.0: Charles Thomas 'Tom' Johnston was born on August 15, 1948.
  Fact 0.1: Charles Thomas 'Tom' Johnston is an American musician.
  Fact 0.2: Charles Thomas Johnston is known by the name 'Tom'.
Sentence 1: Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.
  Fact 1.0: Daryl Palumbo was born on February 10, 1979.
  Fact 1.1: Daryl Palumbo is an American musician.
  Fact 1.2: Daryl Palumbo is originally from Elmira, New York.

Retrieved items:
<item 1>
Daryl Palumbo
Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.

 He is the frontman of the bands Glassjaw, Head Automatica and Color Film.

 As a youth he was a member of the Long Island straight edge band "XbustedX."

 In 1993 he met guitarist Justin Beck, who would become his good friend.

 Together they formed Glassjaw.
</item 1>

<item 2>
Head Automatica
Head Automatica is an American rock band from Long Island, New York fronted by Daryl Palumbo (also of Glassjaw).
</item 2>

<item 3>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 3>

<item 4>
Glassjaw
Glassjaw is an American post-hardcore band from Long Island, New York.

 The band is fronted by vocalist Daryl Palumbo and guitarist Justin Beck.

 The band has been influential in the progression of the underground music scene in the eastern United States and United Kingdom and on the post-hardcore genre,
and are known for their intense live shows and frequent line-up changes.
</item 4>

<item 5>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 5>

<item 6>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 6>

<item 7>
United Nations (band)
The idea of the side project can be traced back to the early 2000s, but it was not until 2008 that United Nations took shape.

 Geoff Rickly has said the band started over tequila with Daryl Palumbo.

 At the start of the band, all members except for Rickly were under contracts with other record labels and were not legally permitted to work in projects
released by another, making it difficult for the official lineup of the band to be well known.

 However some members from the band's original line-up have been revealed through interviews and social media posts, such as Daryl Palumbo, Jonah Bayer and
Lukas Previn.

 Since Rickly was the only member whose primary band was not under contract, his name is the only one that could legally appear in press materials.

 Press photos of the band released in 2008 depict four people wearing Ronald Reagan masks.
</item 7>

<item 8>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 8>

<item 9>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 9>

<item 10>
Tom Johnston
Tom Johnston, Thomas Johnston or Tommy Johnston may refer to:
</item 10>
=== END ITEM id=5eb8156b9d7faec3 ===

=== ITEM id=6b761ca1b87aba63 ===
Gold sentences and facts:
Sentence 0: Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
  Fact 0.0: Marc Rosset was born on 7 November 1970.
  Fact 0.1: Marc Rosset is a former professional tennis player.
  Fact 0.2: Marc Rosset is from Switzerland.
  Fact 0.3: Marc Rosset is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
Sentence 1: Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional tennis player.
  Fact 1.0: Maksim Mirnyi, also known as Max Mirnyi, was born on 6 July 1977 in Minsk.
  Fact 1.1: Maksim Mirnyi's Belarusian name is Максім Мікалаевіч Мірны.
  Fact 1.2: Maksim Mirnyi's Russian name is Максим Николаевич Мирный.
  Fact 1.3: Maksim Mirnyi is a Belarusian professional tennis player.

Retrieved items:
<item 1>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 1>

<item 2>
2011 ATP World Tour Finals
The 2011 ATP World Tour Finals (also known as the 2011 Barclays ATP World Tour Finals for sponsorship reasons) was a tennis tournament that was played at the O
Arena in London, United Kingdom between 20 and 27 November 2011.

 The defending champion in singles was world no. 4 Roger Federer, while the defending champions in doubles were Daniel Nestor and Nenad Zimonjić.

 However, they did not defend their title together as they separated after the 2010 event.

 Zimonjic formed a team with Michaël Llodra for the season, and Nestor partnered with Max Mirnyi.

 Federer successfully defended his crown, winning an unprecedented sixth title, while Nestor and Mirnyi captured the doubles title.
</item 2>

<item 3>
2000 AXA Cup
The 2000 AXA Cup was a men's tennis tournament played on indoor hard courts in London and was part of the International Series Gold of the 2000 ATP Tour.

 It was the 23rd edition of the tournament and ran from February 21 through February 27, 2000.

 Marc Rosset won the singles title.
</item 3>

<item 4>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 4>

<item 5>
2004 Canada Masters – Doubles
The 2004 Canada Masters – Doubles was the men's singles event of the one hundred and fifteenth edition of the Canada Masters; a WTA Tier I tournament and the
most prestigious men's tennis tournament held in Canada.

 Mahesh Bhupathi and Max Mirnyi were the defending champions.

 They were both present but did not compete together.

 Mirnyi partnered with Jonas Björkman, but Bhupathi and partner Leander Paes defeated them 6–4, 6–2, in the final.
</item 5>

<item 6>
1995 Gerry Weber Open
The 1995 Gerry Weber Open was a men's tennis tournament played on outdoor grass courts.

 It was the 3rd edition of the Gerry Weber Open, and was part of the World Series of the 1995 ATP Tour.

 It took place at the Gerry Weber Stadion in Halle, North Rhine-Westphalia, Germany, from 19 June through 26 June 1995.

 Marc Rosset won the singles title.
</item 6>

<item 7>
2010 Australian Open – Mixed Doubles
Sania Mirza and Mahesh Bhupathi were the defending champions.

 They accepted the invitation for the tournament, but withdrew before their match against Carly Gullickson and Bernard Tomic and were replaced by Akgul
Amanmuradova and Rik de Voest.
</item 7>

<item 8>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 8>

<item 9>
Max Mirnyi
Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional
tennis player.
</item 9>

<item 10>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 10>
=== END ITEM id=6b761ca1b87aba63 ===

=== ITEM id=1a824a2888be5ea6 ===
Gold sentences and facts:
Sentence 0:  It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
  Fact 0.0: The series was created by Jay Lender.
  Fact 0.1: The series was created by Micah Wright.
  Fact 0.2: Jay Lender was a former writer for The Angry Beavers.
  Fact 0.3: Jay Lender was a former writer for SpongeBob SquarePants.
  Fact 0.4: Jay Lender was a former writer for Phineas and Ferb.
  Fact 0.5: Micah Wright was a former writer for The Angry Beavers.
  Fact 0.6: Micah Wright was a former writer for SpongeBob SquarePants.
  Fact 0.7: Micah Wright was a former writer for Phineas and Ferb.
Sentence 1:  The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.
  Fact 1.0: The series chronicles the adventures and endeavors of the title character and his various friends.
  Fact 1.1: The series is set in the fictional underwater city of Bikini Bottom.

Retrieved items:
<item 1>
List of SpongeBob SquarePants episodes
"SpongeBob SquarePants" is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series is set in the fictional underwater city of Bikini Bottom, and centers on the adventures and endeavors of SpongeBob SquarePants, an over-optimistic
sea sponge that annoys other characters.

 Many of the ideas for the show originated in an unpublished, educational comic book titled "The Intertidal Zone", which Hillenburg created in the mid-1980s.

 He began developing "SpongeBob SquarePants" into a television series in 1996 after the cancellation of "Rocko's Modern Life", another Nickelodeon television
series which Hillenburg previously directed.
</item 1>

<item 2>
Rock Bottom (SpongeBob SquarePants)
"Rock Bottom" is the 34th episode of the first season of the American animated television series "SpongeBob SquarePants".

 It originally was produced in 1999 and aired on Nickelodeon in the United States on March 15, 2000.

 The series follows the adventures of the title character in the underwater city of Bikini Bottom.

 In the episode, SpongeBob becomes stranded in a abyssopelagic zone called Rock Bottom.
</item 2>

<item 3>
SpongeBob SquarePants (season 3)
The third season of the American animated television series "SpongeBob SquarePants", created by Stephen Hillenburg, aired on Nickelodeon from October 5, 2001 to
October 11, 2004, and consists of 20 episodes.

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, who also acted as the showrunner.

 Hillenburg halted production on the show to work on the 2004 film adaptation of the series, "The SpongeBob SquarePants Movie".

 After production on the film, Hillenburg resigned from the show as its showrunner, and appointed staff writer Paul Tibbitt to overtake the position.

 Season 3 was originally set to end the series after the release of the film, but the success prevented the series from ending, leading to a fourth season.
</item 3>

<item 4>
SpongeBob, You're Fired
"SpongeBob, You're Fired" is a television special of the American animated television series "SpongeBob SquarePants".

 It serves as the 11th episode of the ninth season and the 189th overall episode.

 The animation directors were the supervising director, Alan Smart and Tom Yasumi, and was written by Marc Ceccarelli, Luke Brookshier, and Mr. Lawrence.

 Ceccarelli and Brookshier also served as storyboard directors.

 The episode originally aired in Greece on July 3, 2013.

 In the United States, it aired on Nickelodeon on November 11.

 In this episode of the series, which follows the adventures and endeavors of the title character and his various friends in the underwater city of Bikini
Bottom, SpongeBob gets fired from the Krusty Krab after Mr. Krabs discovers he can save a nickel by eliminating him.

 SpongeBob subsequently decides to apply at other restaurants.
</item 4>

<item 5>
SpongeBob SquarePants: Battle for Bikini Bottom
SpongeBob SquarePants: Battle for Bikini Bottom is a 2003 video game based on the animated series of the same name, developed by Heavy Iron Studios, AWE Games,
and Vicarious Visions, and published by THQ.

 The game was released for the PlayStation 2, Game Boy Advance (GBA), Xbox, and Nintendo GameCube consoles as well as a port for Microsoft Windows.

 The Microsoft Windows version of the game is a point-and-click adventure game, whereas the 6th generation platforms received a 3D platformer.

 The Game Boy Advance version received a 2D platformer.

 All versions of the game feature an original storyline, in which the player attempts to defend Bikini Bottom from an invasion of robots created by Plankton
with a machine called the Duplicatotron 3000, playing as SpongeBob in all versions, as well as Patrick and Sandy in the console versions.

 The game was released on October 31, 2003, in North America and in Europe on November 28, 2003.
</item 5>

<item 6>
Band Geeks
"Band Geeks" is the second part of the 15th episode of the second season, and the second half of the 35th episode overall, of the American animated television
program "SpongeBob SquarePants".

 It originally aired on Nickelodeon in the United States on September 7, 2001.

 It was written by C. H. Greenblatt, Aaron Springer, and Merriwether Williams, and the animation was directed by Frank Weiss.

 Springer served as storyboard director, and Greenblatt served as storyboard artist.

 The song "Sweet Victory" by David Glen Eisley was featured in the episode and was later released on the album "" in 2005.
</item 6>

<item 7>
Robocalypse
Robocalypse is a real-time strategy video game developed and published by Vogster Entertainment for the Nintendo DS handheld video game console.

 It has only been released in North America.

 It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
</item 7>

<item 8>
SpongeBob SquarePants (season 4)
The fourth season of the American animated television series "SpongeBob SquarePants", created by former marine biologist and animator Stephen Hillenburg, aired
on Nickelodeon from May 6, 2005 to July 24, 2007, and contained 20 episodes, beginning with the episodes "Fear of a Krabby Patty" and "Shell of a Man".

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, while writer Paul Tibbitt acted as the supervising/co-executive producer and showrunner.

 The show underwent a hiatus on television as Hillenburg halted the production in 2002 to work on the film adaptation of the series, "The SpongeBob SquarePants
Movie".

 Once the film was finalized and the previous season had completed broadcast on television, Hillenburg wanted to end the show, but the success of the series led
to more episodes, so Tibbitt took over Hillenburg's position as showrunner and began working on a fourth season for broadcast in 2005.
</item 8>

<item 9>
SpongeBob SquarePants
SpongeBob SquarePants is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The series' popularity has made it a media franchise, as well as the highest rated series to ever air on Nickelodeon, and the most distributed property of MTV
Networks.

 As of 2015, the media franchise has generated $12 billion in merchandising revenue for Nickelodeon.
</item 9>

<item 10>
Legends of Bikini Bottom
Legends of Bikini Bottom is an anthology series of six episodes in the American animated television series "SpongeBob SquarePants", as part of its seventh
season.

 As the name suggests, the episodes have plots involving things like monsters, magic and mythical creatures.

 The first episode called "Trenchbillies" first premiered on the online social networking service Facebook on January 27, 2011 before airing on Nickelodeon on
January 29, 2011.

 The four other episodes titled "Sponge-Cano!"

, "The Main Drain", "The Monster Who Came to Bikini Bottom" and "Welcome to the Bikini Bottom Triangle" premiered on Nickelodeon as part of an hour-long special
on January 28, 2011.

 The decision to air the series online was aimed at attracting "the young and the restless hooked to the internet and the social media."
</item 10>
=== END ITEM id=1a824a2888be5ea6 ===

=== ITEM id=104bf6c87f07fc25 ===
Gold sentences and facts:
Sentence 0:  Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.
  Fact 0.0: Watkins was born in Des Moines, Iowa.
  Fact 0.1: Watkins rose to fame in the early 1990s.
  Fact 0.2: Watkins rose to fame as a member of the girl-group TLC.
Sentence 1:  She is best known as the former main vocalist and member of South Korean girl group T-ara.
  Fact 1.0: Watkins is best known as the former main vocalist of the South Korean girl group T-ara.
  Fact 1.1: Watkins is best known as the former member of the South Korean girl group T-ara.

Retrieved items:
<item 1>
Rain (Taeyeon song)
"Rain" is a song by South Korean singer Kim Tae-yeon, a member of the South Korean girl group Girls' Generation.

 It was released digitally by S.M. Entertainment on February 3, 2016.
</item 1>

<item 2>
TLC (group)
TLC is an American girl group whose original line-up consisted of Tionne "T-Boz" Watkins, Lisa "Left Eye" Lopes and Rozonda "Chilli" Thomas.

 Formed in Atlanta, Georgia in 1990, the group was very successful during the 1990s despite numerous spats with the law, each other, and the group's record
label and management.

 They scored nine top-ten hits on the "Billboard" Hot 100, including four number-one singles "Creep", "Waterfalls", "No Scrubs", and "Unpretty".

 The group also recorded four multi-platinum albums, including "CrazySexyCool" (1994) which still remains the only album by a female group to receive a diamond
certification from the Recording Industry Association of America (RIAA).

 TLC also became the first R&B group in history to receive Million certification from the Recording Industry Association of Japan (RIAJ) for "FanMail" (1999).
</item 2>

<item 3>
Tionne Watkins
Tionne Tenese Watkins (born April 26, 1970), better known by her stage name T-Boz, is an American singer, songwriter, actress, author, and executive producer.

 Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.

 She has won four Grammy Awards for her work with TLC.
</item 3>

<item 4>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 4>

<item 5>
3rd APAN Star Awards
The 3rd APAN Star Awards () is an awards ceremony for excellence in television in South Korea.

 It was held at the Hall of Jeongsimhwa International Cultural Center, Chungnam National University in Daejeon on November 15, 2014 and hosted by T-ara's Park
So-yeon and Kim Sung-joo.

 The nominees were chosen from 87 Korean dramas that aired from November 1, 2013 to September 2014.
</item 5>

<item 6>
Kim Tae-yeon (actress)
Kim Tae-yeon (born January 3, 1976) is a South Korean actress.

 She began her entertainment career as a model, winning Model Line's 40th Fashion Model contest in 1996 and the Pantene Model contest sponsored by Ford Models
in 2000.

 Kim made her film debut in the highly controversial film "Lies" in 1999.
</item 6>

<item 7>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 7>

<item 8>
R U the Girl
R U the Girl (also known as R U the Girl with T-Boz and Chilli) is an American reality series that aired on UPN in 2005.

 The series featured Tionne "T-Boz" Watkins and Rozonda "Chilli" Thomas, the remaining members of the all-girl R&B group TLC whose former member, Lisa "Left
Eye" Lopes, died in a car crash in Honduras on April 25, 2002.

 Initially promoted as a contest to permanently replace Lopes 3 years after her death by TLC themselves, both Watkins and Thomas admitted that the winner of the
contest would not be joining TLC full-time and would not be a full-time replacement member; the winner would only provide guest vocals on a new single by the
duo.
</item 8>

<item 9>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 9>

<item 10>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 10>
=== END ITEM id=104bf6c87f07fc25 ===
