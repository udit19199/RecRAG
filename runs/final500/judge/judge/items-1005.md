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

This file has 2 items.

=== ITEM id=84ed308a7c15635d ===
Gold sentences and facts:
Sentence 0: Health or vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually rewarded in some way .
  Fact 0.0: Health or vitality is an attribute assigned to entities in a role-playing or video game.
  Fact 0.1: Health or vitality is assigned to entities such as the player character, enemies, and objects.
  Fact 0.2: Health or vitality indicates the state of an entity in combat.
  Fact 0.3: Health is usually measured in hit points or health points.
  Fact 0.4: Hit points or health points are shortened to HP.
  Fact 0.5: When the HP of a player character reaches zero, the player may lose a life.
  Fact 0.6: When the HP of a player character reaches zero, the character might become incapacitated or die.
  Fact 0.7: When the HP of an enemy reaches zero, the enemy may be defeated or die.
  Fact 0.8: When the HP of an enemy reaches zero, the player is usually rewarded in some way.

Retrieved items:
<item 1>
Harry Potter (film series)
19 November 2010 . Retrieved 30 September 2012 . Jump up ^ `` A Wizard Comes Of Age '' . Retrieved 29 September 2012 . ^ Jump up to : `` Kodak Celebrates the
Oscars ® Feature : Harry Potter and the Deathly Hallows -- Part 1 '' . Kodak . Archived from the original on 25 May 2013 . Retrieved 29 September 2012 . Jump up
^ `` Deathly Hallows to Be Shot Using `` Loads of Hand - Held Cameras , '' Tom Felton Talks Sectumsempra in Half - Blood Prince `` . The Leaky Cauldron. 31
March 2009 . Retrieved 31 March 2009 . Jump up ^ `` Lensers are n't afraid of the dark '' . Variety . 12 December 2012 . Retrieved 30 September 2012 . What I
loved about the last film is that David pushed me to go dark , which all cinematographers love to do . And usually you 're fighting with the producers ( about
the look ) but they all wanted it dark and atmospheric , too .
</item 1>

<item 2>
Harry Potter (film series)
Campaignlive.co.uk . Retrieved 19 October 2017 . ^ Jump up to : Dargis , Manohla ; Scott , A.O. ( 15 July 2007 ) . `` Harry Potter and the Four Directors '' .
The New York Times . Retrieved 29 July 2011 . ^ Jump up to : `` Harry Potter at Leavesden '' . WB Studio Tour . Archived from the original on 10 February 2014 .
Retrieved 16 September 2012 . Jump up ^ `` Warner Bros. Plans Two - Part Film Adaptation of `` Harry Potter and the Deathly Hallows '' to Be Directed by David
Yates `` . Business Wire. 13 March 2008 . Retrieved 6 September 2012 ... expand the screen adaptation of Harry Potter and the Deathly Hallows and release the
film in two parts . Jump up ^ Boucher , Geoff ; Eller , Claudia ( 7 November 2010 ) . `` The end nears for ' Harry Potter ' on film '' . Los Angeles Times .
Retrieved 3 January 2010 .
</item 2>

<item 3>
Harry Potter (film series)
The fifth film , Harry Potter and the Order of the Phoenix , was released by Warner Bros. on 11 July 2007 in English - speaking countries , except for the UK
and Ireland , which released the movie on 12 July . The sixth , Harry Potter and the Half - Blood Prince , was released on 15 July 2009 to critical acclaim and
finished its theatrical run ranked as the number two grossing film of 2009 on the worldwide charts . The final novel , Harry Potter and the Deathly Hallows ,
was split into two cinematic parts : Part 1 was released on 19 November 2010 , and Part 2 , the conclusion to both the final film and the series , was released
on 15 July 2011 . Part 1 was originally scheduled to be released in 3D and 2D , but due to a delay in the 3D conversion process , Warner Bros. released the film
only in 2D and IMAX cinemas . However , Part 2 was released in 2D and 3D cinemas as originally planned .
</item 3>

<item 4>
Harry Potter (film series)
In 2006 , Nicholas Hooper started work on the score to Order of the Phoenix by reuniting with director David Yates . Hooper also composed the soundtrack to Half
- Blood Prince but decided not to return for the final films . In January 2010 , Alexandre Desplat was confirmed to compose the score for Harry Potter and the
Deathly Hallows -- Part 1 . The film 's orchestration started in the summer with Conrad Pope , the orchestrator on the first three Harry Potter films ,
collaborating with Desplat . Pope commented that the music `` reminds one of the old days . '' Desplat returned to score Harry Potter and the Deathly Hallows --
Part 2 in 2011 . Director David Yates stated that he wanted John Williams to return to the series for the final installment , but their schedules did not align
due to the urgent demand for a rough cut of the film . The final recording sessions of Harry Potter took place on 27 May 2011 at Abbey Road Studios with the
London Symphony Orchestra , orchestrator Conrad Pope , and composer Alexandre Desplat .
</item 4>

<item 5>
Health (gaming)
In video games , health can also be displayed graphically , such as with a bar that empties itself when an entity loses health ( a health bar , typically red )
, icons that are `` chipped away '' , or in more novel ways . Contents ( hide ) 1 History 2 Usage 2.1 Regeneration 3 Display 4 References History ( edit )
Dungeons & Dragons co-creator Dave Arneson described the origin of hit points in a 2002 interview . When Arneson was adapting the medieval wargame Chainmail (
1971 ) to a fantasy setting , a process that with Gary Gygax would lead to Dungeons & Dragons , he saw that the emphasis of the gameplay was moving from large
armies to small groups of heroes and eventually to the identification of one player and one character that is essential to role - playing as it was originally
conceived . Players became attached to their heroes and did not want them to die every time they lost a die roll . Players were thus given multiple hit points
which were incrementally decreased as they took damage . Arneson took the concept , along with armor class , from a set of a naval American Civil War game 's
rules .
</item 5>

<item 6>
Health (gaming)
US Navy used similar concept in their tactical war games already in 1920s and 1930s . In their simulation each ship had Life parameter . The unit of Life of the
ship was a number of `` equivalent penetrative 14 - inch shell hits '' . The Navy considered ie. that Kongō - class battlecruiser had 12 Life points and Nagato
- class battleship had 18.8 . A visual power meter representing stamina was used in Nintendo 's 1983 arcade game Punch - Out ! ! and Data East 's 1981 DECO
Cassette System arcade game Flash Boy . Usage ( edit ) In action video games as well as in role - playing games , health points can usually be depleted by
attacking the entity . A defense attribute might reduce the amount of HP that is lost when a character is damaged . It is common in role - playing games for a
character 's maximum health and defense attributes to be gradually raised as the character levels up . In game design , it is deemed important that a player is
aware of it when they are losing health , each hit playing a clear sound effect .
</item 6>

<item 7>
Harry Potter (film series)
Harry Potter and the Deathly Hallows -- Part 2 ( 2011 ) Main article : Harry Potter and the Deathly Hallows -- Part 2 After destroying one Horcrux and
discovering the significance of the three Deathly Hallows , Harry , Ron and Hermione continue to seek the other Horcruxes in an attempt to destroy Voldemort ,
who has now obtained the powerful Elder Wand . The Dark Lord discovers Harry 's hunt for Horcruxes and launches an attack on Hogwarts School , where the trio
return for one last stand against the dark forces that threaten the Wizarding and Muggle worlds . Release The rights for the first four novels in the series
were sold to Warner Bros. for £ 1,000,000 by J.K. Rowling . After the release of the fourth book in July 2000 , the first film , Harry Potter and the
Philosopher 's Stone , was released on 16 November 2001 . In its opening weekend , the film grossed $90 million in the United States alone , which set a record
opening worldwide . The succeeding three motion picture adaptations followed suit in financial success , while garnering positive reviews from fans and critics
.
</item 7>

<item 8>
Health (gaming)
Health ( Gaming ) - wikipedia Health ( Gaming ) Jump to : navigation , search A health bar , a possible representation of the health of a character . Health or
vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its
state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may
lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually
rewarded in some way . Any entity within a game could have a health value , including the player character , non-player characters and objects . Indestructible
entities have no diminishable health value . Health might be displayed as a numeric value , such as `` 50 / 100 '' . Here , the first number indicates the
current amount of HP an entity has and the second number indicates the entity 's maximum HP .
</item 8>

<item 9>
Harry Potter (film series)
Extraordinary images , extraordinary talent '' . Framestore . Jump up ^ `` Cinesite , HP 1 -- 7 '' . Cinesite.com . Archived from the original on 18 December
2010 . Retrieved 2 March 2011 . Jump up ^ `` June 2010 Filming completed '' . Snitchseeker.com. 12 June 2010 . Retrieved 2 March 2011 . Jump up ^ `` 10 Years
Filming -- 24Dec2010 '' . Snitchseeker.com. 21 December 2010 . Retrieved 2 March 2011 . Jump up ^ MuggleNet ( 21 December 2010 ) . `` JK Title reveal '' .
Mugglenet.com . Retrieved 2 March 2011 . Jump up ^ `` Harry Potter and the Order of the Phoenix ( 2007 ) '' . IMDb.com . Jump up ^ `` WB Name Drops Big Titles
'' . ERCBoxOffice. 23 February 2009 . Archived from the original on 26 February 2009 .
</item 9>

<item 10>
Harry Potter (film series)
Jump up ^ MuggleNet ( 19 January 2010 ) . `` CONFIRMED -- Desplat for DH '' . Mugglenet.com . Retrieved 2 March 2011 . Jump up ^ `` Pope on Desplat 's HP7 Pt. 1
Score '' . Snitchseeker.com. 28 August 2010 . Retrieved 2 March 2011 . Jump up ^ `` Alexandre Desplat -- Composer of Part 1 and 2 ( NOTE : Click `` About the
Movie '' , then `` Filmmakers '' , then `` Alexandre Desplat '' ) `` . Harrypotter.warnerbros.com . Retrieved 2 March 2011 . Jump up ^ MuggleNet ( 12 November
2010 ) . `` Yates on Williams , Part 2 '' . Mugglenet.com . Retrieved 2 March 2011 . Jump up ^ `` Conrad Pope '' . www.facebook.com . Jump up ^ `` Framestore -
We are Framestore . Extraordinary images , extraordinary talent '' . Framestore .
</item 10>
=== END ITEM id=84ed308a7c15635d ===

=== ITEM id=14c49ed84c72faa6 ===
Gold sentences and facts:
Sentence 0: Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24 Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5 Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia : 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1
  Fact 0.0: Spain is listed with the value 430.
  Fact 0.1: France is listed with the value 128.
  Fact 0.2: Italy is listed with the value 104.
  Fact 0.3: Russia is listed with the value 100.
  Fact 0.4: Portugal is listed with the value 87.
  Fact 0.5: Germany is listed with the value 77.
  Fact 0.6: United Kingdom is listed with the value 64.
  Fact 0.7: Poland is listed with the value 47.
  Fact 0.8: Greece is listed with the value 46.
  Fact 0.9: Turkey is listed with the value 41.
  Fact 0.10: Belgium is listed with the value 31.
  Fact 0.11: Netherlands is listed with the value 28.
  Fact 0.12: Romania is listed with the value 24.
  Fact 0.13: Switzerland is listed with the value 20.
  Fact 0.14: Austria is listed with the value 14.
  Fact 0.15: Sweden is listed with the value 12.
  Fact 0.16: Croatia is listed with the value 10.
  Fact 0.17: Ireland is listed with the value 9.
  Fact 0.18: Ukraine is listed with the value 9.
  Fact 0.19: Hungary is listed with the value 8.
  Fact 0.20: Bulgaria is listed with the value 6.
  Fact 0.21: Cyprus is listed with the value 6.
  Fact 0.22: Czech Republic is listed with the value 6.
  Fact 0.23: Finland is listed with the value 5.
  Fact 0.24: Lithuania is listed with the value 5.
  Fact 0.25: Norway is listed with the value 5.
  Fact 0.26: Serbia is listed with the value 5.
  Fact 0.27: Slovenia is listed with the value 5.
  Fact 0.28: Denmark is listed with the value 4.
  Fact 0.29: Latvia is listed with the value 4.
  Fact 0.30: Estonia is listed with the value 3.
  Fact 0.31: Azerbaijan is listed with the value 3.
  Fact 0.32: Bosnia and Herzegovina is listed with the value 3.
  Fact 0.33: Georgia is listed with the value 3.
  Fact 0.34: Luxembourg is listed with the value 3.
  Fact 0.35: Slovakia is listed with the value 3.
  Fact 0.36: Armenia is listed with the value 2.
  Fact 0.37: Albania is listed with the value 1.
  Fact 0.38: Andorra is listed with the value 1.
  Fact 0.39: Belarus is listed with the value 1.
  Fact 0.40: Iceland is listed with the value 1.
  Fact 0.41: Macedonia is listed with the value 1.
  Fact 0.42: Malta is listed with the value 1.
  Fact 0.43: Monaco is listed with the value 1.
  Fact 0.44: Montenegro is listed with the value 1.
Sentence 1: United Kingdom : 64
  Fact 1.0: United Kingdom is listed with the value 64.
Sentence 2: Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32 Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20 Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait : 5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24 Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5 Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia : 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1
  Fact 2.0: South Africa is listed under Africa with the value 9.
  Fact 2.1: Egypt is listed under Africa with the value 5.
  Fact 2.2: Morocco is listed under Africa with the value 4.
  Fact 2.3: Tunisia is listed under Africa with the value 3.
  Fact 2.4: Algeria is listed under Africa with the value 1.
  Fact 2.5: United States is listed under Americas with the value 85.
  Fact 2.6: Mexico is listed under Americas with the value 84.
  Fact 2.7: Brazil is listed under Americas with the value 56.
  Fact 2.8: Canada is listed under Americas with the value 32.
  Fact 2.9: Colombia is listed under Americas with the value 14.
  Fact 2.10: Argentina is listed under Americas with the value 10.
  Fact 2.11: Chile is listed under Americas with the value 9.
  Fact 2.12: Venezuela is listed under Americas with the value 8.
  Fact 2.13: Peru is listed under Americas with the value 4.
  Fact 2.14: Puerto Rico is listed under Americas with the value 4.
  Fact 2.15: Dominican Republic is listed under Americas with the value 3.
  Fact 2.16: Guatemala is listed under Americas with the value 3.
  Fact 2.17: Panama is listed under Americas with the value 3.
  Fact 2.18: Costa Rica is listed under Americas with the value 2.
  Fact 2.19: Ecuador is listed under Americas with the value 2.
  Fact 2.20: El Salvador is listed under Americas with the value 2.
  Fact 2.21: Honduras is listed under Americas with the value 2.
  Fact 2.22: Uruguay is listed under Americas with the value 2.
  Fact 2.23: Aruba is listed under Americas with the value 1.
  Fact 2.24: Nicaragua is listed under Americas with the value 1.
  Fact 2.25: Paraguay is listed under Americas with the value 1.
  Fact 2.26: China is listed under Asia with the value 200.
  Fact 2.27: Japan is listed under Asia with the value 99.
  Fact 2.28: South Korea is listed under Asia with the value 43.
  Fact 2.29: Saudi Arabia is listed under Asia with the value 39.
  Fact 2.30: Israel is listed under Asia with the value 24.
  Fact 2.31: India is listed under Asia with the value 20.
  Fact 2.32: Indonesia is listed under Asia with the value 17.
  Fact 2.33: Hong Kong is listed under Asia with the value 14.
  Fact 2.34: Thailand is listed under Asia with the value 11.
  Fact 2.35: United Arab Emirates is listed under Asia with the value 11.
  Fact 2.36: Malaysia is listed under Asia with the value 10.
  Fact 2.37: Singapore is listed under Asia with the value 9.
  Fact 2.38: Taiwan is listed under Asia with the value 9.
  Fact 2.39: Philippines is listed under Asia with the value 8.
  Fact 2.40: Lebanon is listed under Asia with the value 7.
  Fact 2.41: Kazakhstan is listed under Asia with the value 5.
  Fact 2.42: Kuwait is listed under Asia with the value 5.
  Fact 2.43: Qatar is listed under Asia with the value 5.
  Fact 2.44: Jordan is listed under Asia with the value 3.
  Fact 2.45: Bahrain is listed under Asia with the value 2.
  Fact 2.46: Macau is listed under Asia with the value 2.
  Fact 2.47: Oman is listed under Asia with the value 1.
  Fact 2.48: Vietnam is listed under Asia with the value 1.
  Fact 2.49: Spain is listed under Europe with the value 430.
  Fact 2.50: France is listed under Europe with the value 128.
  Fact 2.51: Italy is listed under Europe with the value 104.
  Fact 2.52: Russia is listed under Europe with the value 100.
  Fact 2.53: Portugal is listed under Europe with the value 87.
  Fact 2.54: Germany is listed under Europe with the value 77.
  Fact 2.55: United Kingdom is listed under Europe with the value 64.
  Fact 2.56: Poland is listed under Europe with the value 47.
  Fact 2.57: Greece is listed under Europe with the value 46.
  Fact 2.58: Turkey is listed under Europe with the value 41.
  Fact 2.59: Belgium is listed under Europe with the value 31.
  Fact 2.60: Netherlands is listed under Europe with the value 28.
  Fact 2.61: Romania is listed under Europe with the value 24.
  Fact 2.62: Switzerland is listed under Europe with the value 20.
  Fact 2.63: Austria is listed under Europe with the value 14.
  Fact 2.64: Sweden is listed under Europe with the value 12.
  Fact 2.65: Croatia is listed under Europe with the value 10.
  Fact 2.66: Ireland is listed under Europe with the value 9.
  Fact 2.67: Ukraine is listed under Europe with the value 9.
  Fact 2.68: Hungary is listed under Europe with the value 8.
  Fact 2.69: Bulgaria is listed under Europe with the value 6.
  Fact 2.70: Cyprus is listed under Europe with the value 6.
  Fact 2.71: Czech Republic is listed under Europe with the value 6.
  Fact 2.72: Finland is listed under Europe with the value 5.
  Fact 2.73: Lithuania is listed under Europe with the value 5.
  Fact 2.74: Norway is listed under Europe with the value 5.
  Fact 2.75: Serbia is listed under Europe with the value 5.
  Fact 2.76: Slovenia is listed under Europe with the value 5.
  Fact 2.77: Denmark is listed under Europe with the value 4.
  Fact 2.78: Latvia is listed under Europe with the value 4.
  Fact 2.79: Estonia is listed under Europe with the value 3.
  Fact 2.80: Azerbaijan is listed under Europe with the value 3.
  Fact 2.81: Bosnia and Herzegovina is listed under Europe with the value 3.
  Fact 2.82: Georgia is listed under Europe with the value 3.
  Fact 2.83: Luxembourg is listed under Europe with the value 3.
  Fact 2.84: Slovakia is listed under Europe with the value 3.
  Fact 2.85: Armenia is listed under Europe with the value 2.
  Fact 2.86: Albania is listed under Europe with the value 1.
  Fact 2.87: Andorra is listed under Europe with the value 1.
  Fact 2.88: Belarus is listed under Europe with the value 1.
  Fact 2.89: Iceland is listed under Europe with the value 1.
  Fact 2.90: Macedonia is listed under Europe with the value 1.
  Fact 2.91: Malta is listed under Europe with the value 1.
  Fact 2.92: Monaco is listed under Europe with the value 1.
  Fact 2.93: Montenegro is listed under Europe with the value 1.
  Fact 2.94: Australia is listed under Oceania with the value 19.
  Fact 2.95: New Zealand is listed under Oceania with the value 1.

Retrieved items:
<item 1>
Zara (retailer)
Ortega opened additional stores throughout Spain . During the 1980s , Ortega changed the design , manufacturing , and distribution process to reduce lead times
and react to new trends in a quicker way , which he called `` instant fashions '' . The improvements included the use of information technologies and using
groups of designers instead of individuals . Zara 's previous logo Expansion ( edit ) In 1988 , the company started its international expansion through Porto ,
Portugal . In 1989 , it entered the United States , and then France in 1990 . During the 1990s , Zara expanded to Mexico ( 1992 ) , Greece , Belgium and Sweden
( 1993 ) . In the early 2000s , Zara opened its first stores in Japan and Singapore ( 2002 ) , Russia and Malaysia ( 2003 ) , China , Morocco , Estonia ,
Hungary and Romania ( 2004 ) , the Philippines , Costa Rica and Indonesia ( 2005 ) , South Korea ( 2008 ) , India ( 2010 ) , and South Africa and Australia (
2011 ) .
</item 1>

<item 2>
Zara (retailer)
When the news was eventually picked up by media outlets , and Inditex , Zara 's owning company , was asked to comment on the issue , Inditex replied that the
items in question have been suspended from sale , and that they are in contact with Bassen 's lawyer to clarify and address the issue . In November 2017 ,
customers shopping at Zara stores in Istanbul , Turkey found handwritten notes purportedly from Turkish workers in the pockets of in - store garments asking
shoppers to pressure Zara into paying them the wages they claim they are owed . Stores ( edit ) There are over 2,100 Zara stores located across 88 countries .
Zara usually selects the best located and most expensive real estate locations in the world to open its flagship stores . Zara 's flagship stores : Fifth Avenue
, New York Oxford Street , London Calle Serrano , Madrid Via del Corso , Rome Champs - Élysées , Paris Oscar Freire Street , Sao Paulo Corso Vittorio Emanuele ,
Milan Nevsky Prospect , Saint Petersburg Shibuya and Ginza , Tokyo Myeongdong in Seoul Flora Fountain in Mumbai The number of Zara stores in each country ,
</item 2>

<item 3>
Zara (retailer)
: 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24
Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5
Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia
: 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1 See also ( edit )
Spain portal Companies portal Fashion portal Bonds ( clothing ) Gap Inc . H&M List of companies of Spain European Retail Round Table References ( edit ) Jump up
^ `` Zara '' . inditex.com .
</item 3>

<item 4>
Zara (retailer)
Zara ( retailer ) - Wikipedia Zara ( retailer ) Jump to : navigation , search Zara España , S.A. Type Sociedad Anónima Industry Retail Founded Arteixo , Galicia
( 24 May 1974 ; 43 years ago ( 1974 - 05 - 24 ) ) Founder Amancio Ortega Headquarters Arteixo , Galicia Number of locations 2266 stores Area served Worldwide
Products Clothing Revenue US $ 15.9 billion ( 2016 ) Parent Inditex Website zara.com Zara ( Spanish : ( ˈθaɾa ) ) is a Galician fast fashion ( clothing and
accessories ) retailer based in Arteixo , Galicia . The company was founded in 1975 by Amancio Ortega and Rosalía Mera . It is the main brand of the Inditex
group , the world 's largest apparel retailer . The fashion group also owns brands such as Massimo Dutti , Pull&Bear , Bershka , Stradivarius , Oysho , Zara
Home , and Uterqüe .
</item 4>

<item 5>
Zara (retailer)
24 April 2005 . Retrieved 8 April 2016 . Jump up ^ `` Spains Zara Opens First Store in Korea '' . The Korea Times . 29 April 2008 . Retrieved 8 April 2016 .
Jump up ^ `` Fashion chain Zara opens its first Indian store '' . BBC . 31 May 2010 . Retrieved 8 April 2016 . Jump up ^ Emma Jordan ( 19 January 2011 ) . ``
Zara to Open in South Africa '' . iFashion . Retrieved 8 April 2016 . Jump up ^ `` Global phenomenon Zara finally opens in Sydney '' . Vogue. 19 April 2011 .
Retrieved 8 April 2016 . Jump up ^ `` Zara launches online boutique '' . fashionunited.com. 6 September 2010 . Retrieved 22 June 2014 . Jump up ^ `` Zara.com
launches in 5 more countries '' . Retrieved 30 May 2015 . Jump up ^ `` Inditex post record profits '' .
</item 5>

<item 6>
Zara (retailer)
Products ( edit ) Zara stores have men 's and women 's clothing , as well as children 's clothing ( Zara Kids ) . Zara 's products are supplied based on
consumer trends . Its highly responsive supply chain ships new products to stores twice a week . After products are designed , they take ten to fifteen days to
reach the stores . All of the clothing is processed through the distribution center in Spain . New items are inspected , sorted , tagged , and loaded into
trucks . In most cases , the clothing is delivered within 48 hours . Zara produces over 450 million items per year . Manufacturing and distribution ( edit )
Zara Almere - Heren The men 's department of a typical Zara store . Almere , Netherlands Zara clothing made in Portugal Zara store in New York City . Zara store
in Brisbane Zara store in Palma de Mallorca ( Spain ) . Zara store in Brussels ( Belgium ) . Reportedly , Zara needs just one week to develop a new product and
get it to stores , compared to the six - month industry average , and launches around 12,000 new designs each year .
</item 6>

<item 7>
Zara (retailer)
20 October 2014 . Graham Ruddick . accessed 5 April 2016 Jump up ^ Zara new season styles : what to buy at the Spanish giant right now . Linda Sharkey. 29 May
2015 . Independent . accessed 5 April 2016 Jump up ^ `` Presencia internacional '' . inditex.com. 30 April 2015 . Retrieved 9 September 2015 . Jump up ^ ``
Localizador de tiendas '' ( in Spanish ) . Zara.com . Retrieved 11 June 2015 . External links ( edit ) Official website Media related to Zara ( clothing ) at
Wikimedia Commons Retrieved from `` https://en.wikipedia.org/w/index.php?title=Zara_(retailer)&oldid=819154896 '' Categories : Clothing brands of Spain Clothing
retailers of Spain Companies based in Galicia Clothing companies established in 1974 Retail companies established in 1974 1974 establishments in Spain Inditex
brands Hidden categories : Pages using citations with accessdate and no URL CS1 Dutch - language sources ( nl ) CS1 Portuguese - language sources ( pt )
</item 7>

<item 8>
Zara (retailer)
Zara as of 2017 manages up to 20 clothing collections a year . Contents ( hide ) 1 History 1.1 Early history 1.2 Expansion 2 Products 3 Manufacturing and
distribution 3.1 Non-toxic clothing 4 Controversy 5 Stores 6 See also 7 References 8 External links History ( edit ) Early history ( edit ) Amancio Ortega
opened the first Zara store in 1975 in downtown A Coruña , Galicia , Spain . Ortega initially named the store Zorba after the classic film Zorba the Greek , but
after learning there was a bar with the same name two blocks away , they rearranged the letters molded for the sign to ' Zara . ' It is believed the extra ' a '
came from an additional set of letters that had been made for the company . The first store featured low - priced lookalike products of popular , higher - end
clothing fashions . Ortega opened additional stores throughout Spain .
</item 8>

<item 9>
Zara (retailer)
On September 2010 , Zara launched its online boutique . The website began in Spain , the UK , Portugal , Italy , Germany and France . In November that same year
, Zara Online extended the service to five more countries : Austria , Ireland , the Netherlands , Belgium and Luxembourg . Online stores began operating in the
United States in 2011 , Russia and Canada in 2013 , and Mexico , Romania , and South Korea in 2014 . . India In 4 october 2017 . Zara introduced the use of RFID
technology in its stores in 2014 . The RFID chips are located in the security tags which are removed from clothing when it is purchased and can be reused . The
chip allows the company to quickly take inventory by detecting radio signals from the RFID tags . When an item is sold , the stockroom is immediately notified
so that the item can be replaced . An item that is not on the shelf can easily be found with the RFID tag . In 2015 , Zara was ranked 30 on Interbrand 's list
of best global brands .
</item 9>

<item 10>
Zara (retailer)
as of 31 November 2017 : Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32
Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El
Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20
Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait :
5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United
</item 10>
=== END ITEM id=14c49ed84c72faa6 ===
