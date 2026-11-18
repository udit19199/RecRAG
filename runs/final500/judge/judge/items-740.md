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

=== ITEM id=491694df9a850585 ===
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
20 October 2014 . Graham Ruddick . accessed 5 April 2016 Jump up ^ Zara new season styles : what to buy at the Spanish giant right now . Linda Sharkey. 29 May
2015 . Independent . accessed 5 April 2016 Jump up ^ `` Presencia internacional '' . inditex.com. 30 April 2015 . Retrieved 9 September 2015 . Jump up ^ ``
Localizador de tiendas '' ( in Spanish ) . Zara.com . Retrieved 11 June 2015 . External links ( edit ) Official website Media related to Zara ( clothing ) at
Wikimedia Commons Retrieved from `` https://en.wikipedia.org/w/index.php?title=Zara_(retailer)&oldid=819154896 '' Categories : Clothing brands of Spain Clothing
retailers of Spain Companies based in Galicia Clothing companies established in 1974 Retail companies established in 1974 1974 establishments in Spain Inditex
brands Hidden categories : Pages using citations with accessdate and no URL CS1 Dutch - language sources ( nl ) CS1 Portuguese - language sources ( pt )
</item 1>

<item 2>
Zara (retailer)
Products ( edit ) Zara stores have men 's and women 's clothing , as well as children 's clothing ( Zara Kids ) . Zara 's products are supplied based on
consumer trends . Its highly responsive supply chain ships new products to stores twice a week . After products are designed , they take ten to fifteen days to
reach the stores . All of the clothing is processed through the distribution center in Spain . New items are inspected , sorted , tagged , and loaded into
trucks . In most cases , the clothing is delivered within 48 hours . Zara produces over 450 million items per year . Manufacturing and distribution ( edit )
Zara Almere - Heren The men 's department of a typical Zara store . Almere , Netherlands Zara clothing made in Portugal Zara store in New York City . Zara store
in Brisbane Zara store in Palma de Mallorca ( Spain ) . Zara store in Brussels ( Belgium ) . Reportedly , Zara needs just one week to develop a new product and
get it to stores , compared to the six - month industry average , and launches around 12,000 new designs each year .
</item 2>

<item 3>
Zara (retailer)
as of 31 November 2017 : Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32
Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El
Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20
Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait :
5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United
</item 3>

<item 4>
Zara (retailer)
On September 2010 , Zara launched its online boutique . The website began in Spain , the UK , Portugal , Italy , Germany and France . In November that same year
, Zara Online extended the service to five more countries : Austria , Ireland , the Netherlands , Belgium and Luxembourg . Online stores began operating in the
United States in 2011 , Russia and Canada in 2013 , and Mexico , Romania , and South Korea in 2014 . . India In 4 october 2017 . Zara introduced the use of RFID
technology in its stores in 2014 . The RFID chips are located in the security tags which are removed from clothing when it is purchased and can be reused . The
chip allows the company to quickly take inventory by detecting radio signals from the RFID tags . When an item is sold , the stockroom is immediately notified
so that the item can be replaced . An item that is not on the shelf can easily be found with the RFID tag . In 2015 , Zara was ranked 30 on Interbrand 's list
of best global brands .
</item 4>

<item 5>
Zara (retailer)
: 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24
Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5
Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia
: 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1 See also ( edit )
Spain portal Companies portal Fashion portal Bonds ( clothing ) Gap Inc . H&M List of companies of Spain European Retail Round Table References ( edit ) Jump up
^ `` Zara '' . inditex.com .
</item 5>

<item 6>
Zara (retailer)
When the news was eventually picked up by media outlets , and Inditex , Zara 's owning company , was asked to comment on the issue , Inditex replied that the
items in question have been suspended from sale , and that they are in contact with Bassen 's lawyer to clarify and address the issue . In November 2017 ,
customers shopping at Zara stores in Istanbul , Turkey found handwritten notes purportedly from Turkish workers in the pockets of in - store garments asking
shoppers to pressure Zara into paying them the wages they claim they are owed . Stores ( edit ) There are over 2,100 Zara stores located across 88 countries .
Zara usually selects the best located and most expensive real estate locations in the world to open its flagship stores . Zara 's flagship stores : Fifth Avenue
, New York Oxford Street , London Calle Serrano , Madrid Via del Corso , Rome Champs - Élysées , Paris Oscar Freire Street , Sao Paulo Corso Vittorio Emanuele ,
Milan Nevsky Prospect , Saint Petersburg Shibuya and Ginza , Tokyo Myeongdong in Seoul Flora Fountain in Mumbai The number of Zara stores in each country ,
</item 6>

<item 7>
Zara (retailer)
Zara ( retailer ) - Wikipedia Zara ( retailer ) Jump to : navigation , search Zara España , S.A. Type Sociedad Anónima Industry Retail Founded Arteixo , Galicia
( 24 May 1974 ; 43 years ago ( 1974 - 05 - 24 ) ) Founder Amancio Ortega Headquarters Arteixo , Galicia Number of locations 2266 stores Area served Worldwide
Products Clothing Revenue US $ 15.9 billion ( 2016 ) Parent Inditex Website zara.com Zara ( Spanish : ( ˈθaɾa ) ) is a Galician fast fashion ( clothing and
accessories ) retailer based in Arteixo , Galicia . The company was founded in 1975 by Amancio Ortega and Rosalía Mera . It is the main brand of the Inditex
group , the world 's largest apparel retailer . The fashion group also owns brands such as Massimo Dutti , Pull&Bear , Bershka , Stradivarius , Oysho , Zara
Home , and Uterqüe .
</item 7>

<item 8>
Zara (retailer)
Ortega opened additional stores throughout Spain . During the 1980s , Ortega changed the design , manufacturing , and distribution process to reduce lead times
and react to new trends in a quicker way , which he called `` instant fashions '' . The improvements included the use of information technologies and using
groups of designers instead of individuals . Zara 's previous logo Expansion ( edit ) In 1988 , the company started its international expansion through Porto ,
Portugal . In 1989 , it entered the United States , and then France in 1990 . During the 1990s , Zara expanded to Mexico ( 1992 ) , Greece , Belgium and Sweden
( 1993 ) . In the early 2000s , Zara opened its first stores in Japan and Singapore ( 2002 ) , Russia and Malaysia ( 2003 ) , China , Morocco , Estonia ,
Hungary and Romania ( 2004 ) , the Philippines , Costa Rica and Indonesia ( 2005 ) , South Korea ( 2008 ) , India ( 2010 ) , and South Africa and Australia (
2011 ) .
</item 8>

<item 9>
Zara (retailer)
24 April 2005 . Retrieved 8 April 2016 . Jump up ^ `` Spains Zara Opens First Store in Korea '' . The Korea Times . 29 April 2008 . Retrieved 8 April 2016 .
Jump up ^ `` Fashion chain Zara opens its first Indian store '' . BBC . 31 May 2010 . Retrieved 8 April 2016 . Jump up ^ Emma Jordan ( 19 January 2011 ) . ``
Zara to Open in South Africa '' . iFashion . Retrieved 8 April 2016 . Jump up ^ `` Global phenomenon Zara finally opens in Sydney '' . Vogue. 19 April 2011 .
Retrieved 8 April 2016 . Jump up ^ `` Zara launches online boutique '' . fashionunited.com. 6 September 2010 . Retrieved 22 June 2014 . Jump up ^ `` Zara.com
launches in 5 more countries '' . Retrieved 30 May 2015 . Jump up ^ `` Inditex post record profits '' .
</item 9>

<item 10>
Zara (retailer)
Zara as of 2017 manages up to 20 clothing collections a year . Contents ( hide ) 1 History 1.1 Early history 1.2 Expansion 2 Products 3 Manufacturing and
distribution 3.1 Non-toxic clothing 4 Controversy 5 Stores 6 See also 7 References 8 External links History ( edit ) Early history ( edit ) Amancio Ortega
opened the first Zara store in 1975 in downtown A Coruña , Galicia , Spain . Ortega initially named the store Zorba after the classic film Zorba the Greek , but
after learning there was a bar with the same name two blocks away , they rearranged the letters molded for the sign to ' Zara . ' It is believed the extra ' a '
came from an additional set of letters that had been made for the company . The first store featured low - priced lookalike products of popular , higher - end
clothing fashions . Ortega opened additional stores throughout Spain .
</item 10>
=== END ITEM id=491694df9a850585 ===

=== ITEM id=98d170c538de8eb0 ===
Gold sentences and facts:
Sentence 0: In August 2015 , the series was reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop other series with Marvel instead .
  Fact 0.0: In August 2015, the series was reinvented as Most Wanted.
  Fact 0.1: The series Most Wanted received a pilot order from ABC in August 2015.
  Fact 0.2: The cast was filled out in early 2016.
  Fact 0.3: Production was completed by late March.
  Fact 0.4: The episode was co-produced by ABC Studios and Marvel Television.
  Fact 0.5: In May 2016, ABC announced that it would not pick up the pilot to series.
  Fact 0.6: In May 2016, ABC announced that it would look to develop other series with Marvel instead.

Retrieved items:
<item 1>
Most Wanted (TV pilot)
^ Jump up to : `` Marvel 's Most Wanted '' . MckennaGrace.com . May 3 , 2016 . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017 . ^ Jump
up to : Frederick , Brittany . `` Who directed Spider - Man : Homecoming ? '' . Fansided . Archived from the original on July 14 , 2017 . Retrieved July 14 ,
2017 . ^ Jump up to : Walsh , S.M. ( June 6 , 2016 ) . `` ' Stitchers ' Season 3 : Meet the Cast '' . Heavy.com . Archived from the original on July 14 , 2017 .
Retrieved July 14 , 2017 . Jump up ^ Daniels , Hunter ( July 23 , 2011 ) . `` Comic - Con : Marvel TV Announces Development Slate ; Includes Live - Action AKA
Jessica Jones ( Alias ) , The Hulk , Cloak And Dagger , Mockingbird , Animated Hulk And The Agents Of S.M.A.S.H. , More '' .
</item 1>

<item 2>
Most Wanted (TV pilot)
The series would have been based on storylines occurring at the end of that second season , but ABC passed on the project . In August 2015 , the series was
reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode
was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop
other series with Marvel instead . Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Main 2.2 Guests 3 Production 3.1 Development 3.2 Writing 3.3 Casting
3.4 Filming 3.5 Marvel Cinematic Universe tie - ins 4 No pick - up and future 5 References 6 External links Premise ( edit ) Ex-spies and ex-spouses Bobbi Morse
and Lance Hunter are on the run trying to uncover a conspiracy against them . With no help from S.H.I.E.L.D.
</item 2>

<item 3>
Most Wanted (TV pilot)
In August 2015 , the spin - off series received new life , reworked as Marvel 's Most Wanted and given a pilot order . Bell and Zbyszewski were again developing
the series , writing the pilot and executive producing . The pair were set to serve as showrunners for the series , if it was picked up . Loeb and Jim Chory
were also attached as executive producers . The series would still focus on Morse and Hunter , with Palicki and Blood both attached , and was described as `` a
new take focusing on the same duo and their continuing adventures . '' In January 2016 , Lee stated that the pilot was `` absolutely '' moving forward ,
praising the script and saying production would begin `` in the next few months . '' In the Agents of S.H.I.E.L.D. episode `` Parting Shot '' , the two
characters were written off that series , having them leave and disavow S.H.I.E.L.D.
</item 3>

<item 4>
Most Wanted (TV pilot)
Retrieved March 25 , 2016 . ^ Jump up to : Abrams , Natalie ( July 25 , 2014 ) . `` Agents of S.H.I.E.L.D. Reveals New Characters , Carter Adds Captain
Directors '' . TV Guide . Archived from the original on July 26 , 2014 . Retrieved July 25 , 2014 . ^ Jump up to : Andreeva , Nellie ( January 26 , 2016 ) . ``
' Marvel 's Most Wanted ' ABC Pilot Casts Delroy Lindo As Dominic Fortune '' . Deadline.com . Archived from the original on January 26 , 2016 . Retrieved
January 27 , 2016 . Jump up ^ Huver , Scott ( April 22 , 2016 ) . `` '' Marvel 's Most Wanted '' Co-Star Andrade Promises An `` Unpredictable '' Series `` .
Comic Book Resources . Archived from the original on April 22 , 2016 . Retrieved April 22 , 2016 . ^ Jump up to : `` Marvel 's Most Wanted '' .
</item 4>

<item 5>
Most Wanted (TV pilot)
At that time , the producers were prepping `` if - come '' orders , `` deals with writers that will come into effect in the event a pilot is picked ( up ) to
series '' . On May 12 , 2016 , the pilot was passed on by ABC , with Lee 's successor Channing Dungey explaining that Most Wanted `` did not feel as strong as
some of the other pilots that we shot . We talked about it with Marvel and we all came to an agreement that we want to figure out what the next show is that we
do together , is something that we all feel is as creatively strong as it can be . '' However , Dungey said that ABC was open to the series being shopped to
another network if Marvel found `` another home '' for it . Loeb also acknowledged this possibility , but said that the series was designed `` to do a very
specific thing '' alongside S.H.I.E.L.D. on ABC . References ( edit ) ^ Jump up to : `` 2016 ABC Pilots '' . Deadline.com . February 22 , 2016 . Archived from
the original on February 23 , 2016 .
</item 5>

<item 6>
Most Wanted (TV pilot)
^ Jump up to : Schwartz , Terri ( March 22 , 2016 ) . `` Agents of SHIELD EP Jeff Bell on Mockingbird and Hunter Departure and Marvel 's Most Wanted '' . IGN .
Archived from the original on March 23 , 2016 . Retrieved March 23 , 2016 . Jump up ^ Schwartz , Terri ( January 9 , 2016 ) . `` Agents of SHIELD Showrunners
Explain Marvel 's Most Wanted Spinoff Plan '' . IGN . Archived from the original on January 10 , 2016 . Retrieved January 10 , 2016 . ^ Jump up to : Lovett ,
Jamie ( October 27 , 2015 ) . `` EXCLUSIVE Agents Of S.H.I.E.L.D. EP Talks Secret Warriors , Civil War , Last Week 's Shocking Death & More '' . ComicBook.com .
Archived from the original on October 27 , 2015 . Retrieved October 28 , 2015 . ^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) .
</item 6>

<item 7>
Most Wanted (TV pilot)
Retrieved August 9 , 2016 . External links ( edit ) Marvel 's Most Wanted on IMDb Marvel Cinematic Universe Feature films Phase One Iron Man The Incredible Hulk
Iron Man 2 Thor Captain America : The First Avenger The Avengers accolades Phase Two Iron Man 3 Thor : The Dark World Captain America : The Winter Soldier
Guardians of the Galaxy Avengers : Age of Ultron Ant - Man Phase Three Captain America : Civil War Doctor Strange Guardians of the Galaxy Vol. 2 Spider - Man :
Homecoming Thor : Ragnarok Black Panther Avengers : Infinity War production Ant - Man and the Wasp Untitled Avengers film production Television series ABC
Agents of S.H.I.E.L.D. episodes season 1 5 characters Agent Carter season 1 characters Inhumans `` Behold ... The Inhumans '' `` Those Who Would Destroy Us ''
Netflix Daredevil season 1 characters Jessica Jones season 1 characters Luke Cage season 1 characters Iron Fist season 1 characters The Defenders The Punisher
Hulu Runaways Freeform Cloak & Dagger Unaired pilot Most Wanted Other Short films Item 47 Agent Carter All Hail the King Digital series WHIH Newsfront Agents of
S.H.I.E.L.D.
</item 7>

<item 8>
Most Wanted (TV pilot)
^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) . `` How Will ' Agents of SHIELD ' Set Up Spinoff ' Marvel 's Most Wanted ' ? '' . The Hollywood Reporter
. Archived from the original on August 20 , 2016 . Retrieved August 20 , 2016 . Jump up ^ Carinacas , Peter ( February 17 , 2016 ) . `` Artisans Booked on '
Marvel 's Most Wanted , ' ' Pearl , ' ' Dirty Dancing ' and ' Stronger ' '' . Variety . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017
. Jump up ^ Carinacas , Peter ( March 15 , 2016 ) . `` Agencies Book Artisans on ' Thor : Ragnarok , ' ' Sugar Queen , ' ' Chagrin Falls ' and ' Bunker Hill '
'' . Variety . Archived from the original on March 20 , 2016 . Retrieved March 20 , 2016 .
</item 8>

<item 9>
Most Wanted (TV pilot)
`` ' Marvel 's Most Wanted ' Not Going Forward at ABC '' . Variety . Archived from the original on May 13 , 2016 . Retrieved May 13 , 2016 . Jump up ^ Topel ,
Fred ( August 4 , 2016 ) . `` Why ABC Passed on ' Marvel 's Most Wanted ' and ' Agent Carter ' Season 3 ( TCA 2016 ) '' . / Film . Archived from the original on
August 5 , 2016 . Retrieved August 6 , 2016 . Jump up ^ Topel , Fred ( August 8 , 2016 ) . `` Marvel TV 's Jeph Loeb on Fox 's `` Reimagined '' ' Hellfire Club
' Series , ' Cloak and Dagger ' , and More ( TCA 2016 ) `` . / Film . Archived from the original on August 9 , 2016 . Retrieved August 9 , 2016 .
</item 9>

<item 10>
Most Wanted (TV pilot)
Most Wanted ( TV pilot ) - wikipedia Most Wanted ( TV pilot ) Jump to : navigation , search Most Wanted Genre Action / Adventure Science fiction Superhero Drama
Created by Jeffrey Bell Paul Zbyszewski Based on Mockingbird by Gerry Conway Barry Smith Lance Hunter by Gary Friedrich Herb Trimpe Starring Adrianne Palicki
Nick Blood Delroy Lindo Fernanda Andrade Oded Fehr Country of origin United States Original language ( s ) English Production Executive producer ( s ) Jeffrey
Bell Paul Zbyszewski Jeph Loeb Jim Chory Production company ( s ) ABC Studios Marvel Television Distributor Disney -- ABC Domestic Television Release Original
network ABC Picture format 720p ( HDTV ) Audio format 5.1 surround sound Chronology Related shows Marvel 's Agents of S.H.I.E.L.D. Marvel Cinematic Universe
television series Marvel 's Most Wanted , or simply Most Wanted , is an unaired American television pilot based on the Marvel Comics characters Bobbi Morse and
Lance Hunter .
</item 10>
=== END ITEM id=98d170c538de8eb0 ===

=== ITEM id=e11bb23d745840eb ===
Gold sentences and facts:
Sentence 0: The Pakistani National Assembly ( Urdu : قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority .
  Fact 0.0: The Pakistani National Assembly is the lower house of the bicameral Majlis-e-Shura.
  Fact 0.1: The Urdu name of the Pakistani National Assembly is Qaumī Asimbli'e Pākistān.
  Fact 0.2: The Majlis-e-Shura also comprises the President of Pakistan.
  Fact 0.3: The Majlis-e-Shura also comprises the Senate, which is the upper house.
  Fact 0.4: The National Assembly convenes at Parliament House in Islamabad.
  Fact 0.5: The Senate convenes at Parliament House in Islamabad.
  Fact 0.6: The National Assembly is a democratically elected body.
  Fact 0.7: The National Assembly has a total of 332 members.
  Fact 0.8: The members of the National Assembly are referred to as Members of the National Assembly (MNAs).
  Fact 0.9: 272 members of the National Assembly are directly elected.
  Fact 0.10: 70 members of the National Assembly hold reserved seats for women and religious minorities.
  Fact 0.11: A political party must secure 172 seats to obtain a majority.
  Fact 0.12: A political party must secure 172 seats to preserve a majority.

Retrieved items:
<item 1>
National Assembly of Pakistan
National Assembly of Pakistan - wikipedia National Assembly of Pakistan Jump to : navigation , search For List of MNAs elected in Pakistani general election ,
2013 , see List of MNAs elected in Pakistani general election , 2013 . National Assembly of Pakistan قومی اسمبلئ پاکستان ‬ 14th National Assembly of Pakistan
Type Type Lower house of the Parliament of Pakistan Leadership Speaker Ayaz Sadiq ( PML - N ) Since 9 November 2015 Deputy Speaker Javed Abbasi ( PML - N )
Since 3 June 2013 Leader of House Shahid Khaqan Abbasi ( PML - N ) Since 1 August 2017 Leader of Opposition Khurshid Shah ( PPP ) Since 7 June 2013 Structure
Seats 332 ( 3 vacant ) Political groups Government Coalition ( 209 ) PML -- N ( 188 ) JUI -- F ( 13 ) PML -- F ( 5 ) NPP ( 2 ) NP ( 1 ) Opposition ( 130 ) PPP (
47 ) PTI ( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP (
</item 1>

<item 2>
National Assembly of Pakistan
The committees have also been empowered to invite or summon before it any member or any other person having a special interest in relation to any matter under
its consideration and may hear expert evidence and hold public hearing . Composition and elections ( edit ) The composition of the National Assembly is
specified in Article 51 of the Constitution of Pakistan . There are a total of 332 seats in the National Assembly . Of these , 272 are filled by direct
elections . In addition , the Pakistani Constitution reserves 10 seats for religious minorities and 60 seats for women , to be filled by proportional
representation among parties with more than 5 % of the vote . As of 2006 , there are 72 women members in the Assembly . Members of the National Assembly are
elected by the people in competitive multi-party elections , to be held at most five years apart on universal adult franchise . To be a member of electoral
college , according to Article 62 of the Constitution , candidates must be citizens of Pakistan and not less than 18 years of age . 2013 election ( edit ) Main
article : Pakistani general election , 2013 Results of the Pakistani general election ,
</item 2>

<item 3>
National Assembly of Pakistan
96 % 27 8 0 0 8 Total 45,388,404 100 % 266 266 60 10 336 Withheld : NA - 103 , NA - 237 ; Postponed : NA - 38 ; Re-poll ordered : NA - 46 ; Terminated : NA - 83
, NA - 254 Source : Detailed Results General Elections - 2013 , Reserve Seats Women , Reserve Seats Minorities Dissolution ( edit ) National Assembly can be
dissolved at the initiative of the Prime Minister . If dissolved , new elections are conducted for the Assembly . Article 58 of the Constitution of Pakistan
deals with the dissolution of the Assembly : 58 . Dissolution of the National Assembly : ( 1 ) The President shall dissolve the National Assembly if so advised
by the Prime Minister ; and the National Assembly shall , unless sooner dissolved , stand dissolved at the expiration of forty - eight hours after the Prime
Minister has so advised .
</item 3>

<item 4>
National Assembly of Pakistan
The Public Accounts Committee has a special role to review the report of the Auditor General . Senate , the upper house of the Parliament , has equal
representation from the federating units balancing the provincial inequality in the National Assembly , where the number of members is based on population of
the provinces . The Senate 's role is to promote national cohesion and harmony , and work as a stabilizing factor of the federation . The Senate numbers a total
of 104 members who serve six - year terms which are alternated so that half the senators are up for re-election by the electoral college every three years . The
National Assembly consists of 332 members . The Constitution empowers the President to dissolve the National Assembly , but the Senate is not subject to
dissolution . Only the Parliament can amend the Constitution , by two - thirds majority vote separately in each House . Members of National Assembly ( edit )
See also : List of Pakistan National Assembly Seats Qualifications ( edit ) The Constitution of Pakistan lists a number of requirements for members of the
National Assembly in Article 62 .
</item 4>

<item 5>
National Assembly of Pakistan
Seat distribution ( edit ) In the 14th Assembly , the seats have been allocated among the four provinces , federally - administered area , and capital territory
of Pakistan as follows : Province General Women Total Punjab 148 35 183 Sindh 61 14 75 KPK 35 8 43 Balochistan 14 17 FATA 12 - 12 Federal Capital - See also (
edit ) Speaker of National Assembly Senate of Pakistan Politics of Pakistan Prime Minister of Pakistan Constitution of Pakistan President of Pakistan Supreme
Court of Pakistan Pakistan Bar Council Punjab Bar Council Supreme Court Bar Association of Pakistan List of Pakistan National Assembly Seats List of provincial
governments of Pakistan showing provincial assemblies Notes ( edit ) ^ Jump up to : `` PML - N 's Ayaz Sadiq elected Speaker , Murtaza Abbasi Deputy Speaker in
NA '' . Dawn News . 3 June 2013 . Retrieved 6 June 2013 . Jump up ^ `` PPP 's Khursheed Shah declared leader of opposition in NA '' . Dawn News . 7 June 2013 .
Retrieved 7 June 2013 .
</item 5>

<item 6>
National Assembly of Pakistan
631 0.47 % 0 9 National Peoples Party 197,829 0.44 % 0 10 Pakistan Muslim League ( Q ) 1,409,905 3.11 % 0 0 11 Awami National Party 453,057 1.00 % 0 0 12 Qaumi
Watan Party ( Sherpao ) 46,574 0.10 % 0 0 13 Pakistan Muslim League ( Z ) 128,510 0.28 % 0 0 14 National Party 61,148 0.13 % 0 0 15 Awami Muslim League Pakistan
93,046 0.20 % 0 0 16 Awami Jamhuri Ittehad Pakistan 71,175 0.16 % 0 0 17 Balochistan National Party 63,979 0.14 % 0 0 18 All Pakistan Muslim League 54,231 0.12
% 0 0 19 Independents 5,880,658 12.96 % 27 8 0 0 8 Total 45,388,
</item 6>

<item 7>
National Assembly of Pakistan
- 26 ( Bannu ) NA - 27 ( Lakki Marwat ) NA - 28 ( Buner ) NA - 29 ( Swat - I ) NA - 30 ( Swat - II ) NA - 31 ( Shangla ) NA - 32 ( Chitral ) NA - 33 ( Upper Dir
- cum - Lower Dir ) NA - 34 ( Lower Dir ) NA - 35 ( Malakand ) National Assembly constituencies from Islamabad and FATA ( NA - 36 to NA - 49 ) NA - 36 ( Tribal
Area - I ) NA - 37 ( Tribal Area - II ) NA - 38 ( Tribal Area - III ) NA - 39 ( Tribal Area - IV ) NA - 40 ( Tribal Area - V ) NA - 41 ( Tribal Area - VI ) NA -
42 ( Tribal Area - VII ) NA - 43 ( Tribal Area - VIII ) NA - 44 ( Tribal Area - IX ) NA - 45 ( Tribal Area - X ) NA - 46 ( Tribal Area - XI ) NA - 47 ( Tribal
Area - XII ) NA - 48 ( Islamabad - I ) NA - 49 ( Islamabad -
</item 7>

<item 8>
National Assembly of Pakistan
( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP ( 3 ) ANP ( 2 ) PML -- Q ( 2 ) AJIP ( 1 ) AML ( 1 ) APML ( 1 ) BNP ( 1 ) PML -- Z ( 1 ) QWP ( 1 ) Ind ( 10 ) Elections Voting
system Mixed member majoritarian ( First past the post for most seats , 60 seats reserved for women and 10 seats reserved for religious minorities by
proportional representation ) Last election 11 May 2013 Meeting place Parliament House , Islamabad Website www.na.gov.pk The Pakistani National Assembly ( Urdu
: قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan
and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad .
</item 8>

<item 9>
National Assembly of Pakistan
The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a
total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for
women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority . Members are elected through the first - past - the
- post system under universal adult suffrage , representing electoral districts known as National Assembly constituencies . According to the constitution , the
70 seats reserved for women and religious minorities are allocated to the political parties according to their proportional representation . Each National
Assembly is formed for a five - year term , commencing from the date of the first sitting , after which it is automatically dissolved . Currently the National
Assembly can not be dissolved by the President of Pakistan , it is dissolved by the Prime Minister of Pakistan . Election for 13th National Assembly was held on
18 February 2008 . On March 17 , 2013 13th National Assembly was dissolved on completion of its five - year term under Article 52 of the Constitution .
</item 9>

<item 10>
National Assembly of Pakistan
Area - XII ) NA - 48 ( Islamabad - I ) NA - 49 ( Islamabad - II ) National Assembly constituencies from the Punjab ( NA - 50 to NA - 197 ) NA - 50 ( Rawalpindi
- I ) NA - 51 ( Rawalpindi - II ) NA - 52 ( Rawalpindi - III ) NA - 53 ( Rawalpindi - IV ) NA - 54 ( Rawalpindi - V ) NA - 55 ( Rawalpindi - VI ) NA - 56 (
Rawalpindi - VII ) NA - 57 ( Attock - I ) NA - 58 ( Attock - II ) NA - 59 ( Attock - III ) NA - 60 ( Chakwal - I ) NA - 61 ( Chakwal - II ) NA - 62 ( Jhelum - I
) NA - 63 ( Jhelum - II ) NA - 64 ( Sargodha - I ) NA - 65 ( Sargodha - II ) NA - 66 ( Sargodha - III ) NA - 67 ( Sargodha - IV ) NA
</item 10>
=== END ITEM id=e11bb23d745840eb ===

=== ITEM id=11b9c24f41778ba6 ===
Gold sentences and facts:
Sentence 0: Loyalists were American colonists who remained loyal to the British Crown during the American Revolutionary War . At the time they were often called Tories , Royalists , or King 's Men . They were opposed by the Patriots , those who supported the revolution and called them `` persons inimical to the liberties of America '' . Prominent Loyalists repeatedly assured the British government that many thousands of loyalists would spring to arms and fight for the crown . The British government acted in expectation of that , especially in the southern campaigns in 1780 - 81 . In practice , the number of loyalists in military service was far lower than expected . Across the colonies , Patriots watched suspected Loyalists very closely , and would not tolerate any organized Loyalist opposition . Many outspoken or militarily active loyalists were forced to flee , especially to their stronghold of New York City . William Franklin , the royal governor of New Jersey and son of Patriot leader Benjamin Franklin , became the leader of the Loyalists after his release from a Patriot prison in 1778 . He worked to build Loyalist military units to fight in the war , but the number of volunteers was much fewer than London expected .
  Fact 0.0: Loyalists were American colonists who remained loyal to the British Crown during the American Revolutionary War.
  Fact 0.1: Loyalists were called Tories, Royalists, or King's Men at the time.
  Fact 0.2: Loyalists were opposed by the Patriots.
  Fact 0.3: The Patriots were those who supported the revolution.
  Fact 0.4: The Patriots called the Loyalists "persons inimical to the liberties of America."
  Fact 0.5: Prominent Loyalists repeatedly assured the British government that many thousands of loyalists would spring to arms and fight for the Crown.
  Fact 0.6: The British government acted in expectation of that, especially in the southern campaigns in 1780-81.
  Fact 0.7: In practice, the number of loyalists in military service was far lower than expected.
  Fact 0.8: Patriots watched suspected Loyalists very closely across the colonies.
  Fact 0.9: Patriots would not tolerate any organized Loyalist opposition.
  Fact 0.10: Many outspoken or militarily active loyalists were forced to flee.
  Fact 0.11: Loyalists fled especially to their stronghold of New York City.
  Fact 0.12: William Franklin was the royal governor of New Jersey.
  Fact 0.13: William Franklin was the son of Patriot leader Benjamin Franklin.
  Fact 0.14: William Franklin became the leader of the Loyalists after his release from a Patriot prison in 1778.
  Fact 0.15: William Franklin worked to build Loyalist military units to fight in the war.
  Fact 0.16: The number of volunteers for William Franklin's Loyalist military units was much fewer than London expected.

Retrieved items:
<item 1>
Loyalist (American Revolution)
Gilbert Stuart painted a portrait of James DeLancey around 1785 . It stays in the collection of the Metropolitan Museum of Art , a bequest of his descendant
George DeLancey Harris , Jr. of New York City & Annapolis Royal , NS . Loyalists in literature ( edit ) See also : Category : American Revolutionary War novels
The Adventures of Jonathan Corncob , Loyal American Refugee ( 1787 ) by Jonathan Corncob . According to Maya Jasanoff , `` traveling to London to file a claim
served as the opening gambit '' for this `` picaresque novel about the American Revolution '' .
</item 1>

<item 2>
Loyalist (American Revolution)
`` Rip Van Winkle '' ( 1819 ) , short story by Washington Irving The Spy : a Tale of the Neutral Ground ( 1821 ) , novel by James Fenimore Cooper Oliver Wiswell
( 1940 ) , a novel by Kenneth Roberts The Book of Negroes ( 2007 ) by Lawrence Hill The Fort ( 2010 ) , novel by Bernard Cornwell Notable Loyalists ( edit )
Samuel Adams ( 1730 -- 1810 ) , Vermont physician ; not related to Samuel Adams the Patriot leader in Boston Rev. H. Addison of Maryland was forbidden to ever
live in his home state after the Revolution . Rev. John Agnew ( d . 1812 , New Brunswick ) served a Church of England parish in Suffolk , Virginia . Samuel
Aikens , Land grant dated 2 June 1785 Guysborough , Nova Scotia , Canada Andrew Allen ( 1740 -- 1825 ) , Pennsylvanian Delegate to the Second Continental
Congress Isaac Allen ( d .
</item 2>

<item 3>
Loyalist (American Revolution)
Most of the English - speaking settlers had arrived following the British conquest of Canada in 1759 -- 1760 , and were unlikely to support separation from
Britain . The older British colonies , Newfoundland and Nova Scotia ( including what is now New Brunswick ) also remained loyal and contributed military forces
in support of the Crown . In late 1775 the Continental Army sent a force into Quebec , led by General Richard Montgomery and Colonel Benedict Arnold , with the
goal of convincing the residents of Quebec to join the Revolution . Although only a minority of Canadians openly expressed loyalty to King George , about 1,500
militia fought for the King in the Siege of Fort St. Jean . In the region south of Montreal that was occupied by the Continentals , some inhabitants supported
the rebellion and raised two regiments to join the Patriot forces . In Nova Scotia , there were many Yankee settlers originally from New England , and they
generally supported the principles of the revolution . This element was declining in relative numbers and influence due to an influx of recent immigration from
the British isles , and they remained neutral during the war , and the influx was greatest in Halifax .
</item 3>

<item 4>
Loyalist (American Revolution)
son of a Taunton , MA Patriot , he later served as the Chief Justice of the Bermudas John Lovell , Headmaster of the Boston Latin School Isaac Low ( 1735 --
1791 ) , delegate for New York to the First Continental Congress in 1774 and to New York Provincial Congress and first President of the New York Chamber of
Commerce Gabriel Ludlow , New York merchant George Ludlow , New York judge David Mathews , Princeton University graduate and Mayor of New York City ; subsequent
administrator of Cape Breton , Nova Scotia William McCormick ( 1742 -- 1815 ) , North Carolina merchant Colonel Alexander McKee ( ca . 1735 -- 1799 ) , liaison
between the British and the Shawnees Benjamin Milliken , founder of Ellsworth Maine , mill and ship owner James Moody , Lieutenant , First New Jersey Volunteers
, March 1781 Benjamin Moore ( 1748 -- 1816 ) , Assistant Rector of Trinity Church , New York City whose son Clement composed `` A Visit from St.
</item 4>

<item 5>
Loyalist (American Revolution)
David Farnsworth ( d . 1778 ) , British agent hanged for his participation in a plot to undermine the American economy by distributing counterfeit currency
Elizabeth Graeme Fergusson , female Loyalist who wrote poems lamenting her husband 's desertion of her . These poems shared the grief that she and other women
left behind were feeling . William Franklin ( ca . 1730 -- 1813 ) , Governor of New Jersey , son of Benjamin Franklin Rebecca Franks ( 1760 -- 1823 ) ,
prominent member of Loyalist society in Philadelphia , Pennsylvania during the American Revolution Shaddrick Furman , a Virginia free black , was a guide and
information source for British troops . Upon his capture by Patriots , he was beaten and blinded . He received a pension of £ 18 per year for life from the
Loyalist Claims Commission .
</item 5>

<item 6>
Loyalist (American Revolution)
George Washington 's Mount Vernon . Jump up ^ `` The Loyalists '' . Jump up ^ Calhoun ( 1973 ) Jump up ^ Georgia Encyclopædia . Jump up ^ Robert M. Calhoon , in
' A companion to the American Revolution ' , ( 2000 ) ; p 235 . Jump up ^ John Adams has sometimes been cited as having claimed , in an 1813 letter , that one -
third of Americans supported the revolution and one - third were against . However , the passage in question actually refers to the French Revolution of 1789 .
Robert D. Marcus ( 1971 ) . The American Scene : Varieties of American History . p. 147 . See also `` Only 1 / 3 of Americans Supported the American Revolution
? '' , by William Marina . 6 - 28 - 2004 . Retrieved on July 14 , 2008 . Jump up ^ Ray Raphael ( 2012 ) . A People 's History of the American Revolution . The
New Press . p. 393 .
</item 6>

<item 7>
Loyalist (American Revolution)
They felt themselves to be weak or threatened within American society and in need of an outside defender such as the British Crown and Parliament . They had
been promised freedom from slavery by the British . They felt that being a part of the British Empire was crucial in terms of commerce and their business
operations . Loyalism and military operations ( edit ) See also : Loyalists fighting in the American Revolution In the opening months of the Revolutionary War ,
the Patriots laid siege to Boston , where most of the British forces were stationed . Elsewhere there were few British troops and the Patriots seized control of
all levels of government , as well as supplies of arms and gunpowder . These actions were not without resistance . Especially in New York , New Jersey , and
parts of North and South Carolina , there was considerable ambivalence about the Patriot cause . Vocal Loyalists , often with the encouragement and assistance
of royal governors , recruited people to their side . In the South Carolina backcountry Loyalist recruitment oustripped that of Patriots .
</item 7>

<item 8>
Loyalist (American Revolution)
Chesterton Winston Churchill Enoch Powell George Grant Related topics ( show ) Conservatism Distributists Miguelism Vendéens Chouans Legitimism Carlism
Sanfedismo Viva Maria Cristeros Reactionary High Tory Red Tory Loyalist Royalist Veronese Easter Tory corporatism Tory socialism Pink Tory Ultra-Tories
Britannia offers solace and a promise of compensation for her exiled American - born Loyalists . ( Reception of the American Loyalists by Great Britain in the
Year 1783 . Engraving by H. Moses after Benjamin West . ) Loyalists were American colonists who remained loyal to the British Crown during the American
Revolutionary War . At the time they were often called Tories , Royalists , or King 's Men . They were opposed by the Patriots , those who supported the
revolution and called them `` persons inimical to the liberties of America '' . Prominent Loyalists repeatedly assured the British government that many
thousands of loyalists would spring to arms and fight for the crown . The British government acted in expectation of that , especially in the southern campaigns
in 1780 - 81 .
</item 8>

<item 9>
Loyalist (American Revolution)
Historian Robert Middlekauff summarized scholarly research on the nature of Loyalist support as follows : The largest number of loyalists were found in the
middle colonies : many tenant farmers of New York supported the king , for example , as did many of the Dutch in the colony and in New Jersey . The Germans in
Pennsylvania tried to stay out of the Revolution , just as many Quakers did , and when that failed , clung to the familiar connection rather than embrace the
new . Highland Scots in the Carolinas , a fair number of Anglican clergy and their parishioners in Connecticut and New York , a few Presbyterians in the
southern colonies , and a large number of the Iroquois Indians stayed loyal to the king . Johnson Hall , seat of Sir John Johnson in the Mohawk Valley New York
City and Long Island were the British military and political base of operations in North America from 1776 to 1783 and had a large concentration of Loyalists ,
many of whom were refugees from other states . According to Calhoon , Loyalists tended to be older and wealthier , but there were also many Loyalists of humble
means .
</item 9>

<item 10>
Pencil
This method of manufacture , which had been earlier discovered by the Austrian Joseph Hardtmuth , the founder of the Koh - I - Noor in 1790 , remains in use .
In 1802 , the production of graphite leads from graphite and clay was patented by the Koh - I - Noor company in Vienna . In England , pencils continued to be
made from whole sawn graphite . Henry Bessemer 's first successful invention ( 1838 ) was a method of compressing graphite powder into solid graphite thus
allowing the waste from sawing to be reused . The pencil in america Pencil , perhaps made by Henry David Thoreau , in the Concord Museum Pencil manufacturing .
The top sequence shows the old method that required pieces of graphite to be cut to size ; the lower sequence is the new , current method using rods of graphite
and clay . American colonists imported pencils from Europe until after the American Revolution . Benjamin Franklin advertised pencils for sale in his
Pennsylvania Gazette in 1729 , and George Washington used a three - inch pencil when he surveyed the Ohio Territory in 1762 .
</item 10>
=== END ITEM id=11b9c24f41778ba6 ===

=== ITEM id=b424633de01528ad ===
Gold sentences and facts:
Sentence 0: The film is based loosely on Eminem 's actual upbringing , and follows white rapper B - Rabbit ( Eminem ) and his attempt to launch a career in a genre dominated by African - Americans . The title is derived from 8 Mile Road , the highway between the predominantly black city of Detroit and Wayne County and the predominantly White Oakland County and Macomb County suburbs . It was filmed mostly on location .
  Fact 0.0: The film is based loosely on Eminem's actual upbringing.
  Fact 0.1: The film follows white rapper B-Rabbit (Eminem).
  Fact 0.2: B-Rabbit (Eminem) attempts to launch a career in a genre dominated by African-Americans.
  Fact 0.3: The title of the film is derived from 8 Mile Road.
  Fact 0.4: 8 Mile Road is the highway between the predominantly black city of Detroit and Wayne County.
  Fact 0.5: 8 Mile Road is the highway between the predominantly White Oakland County and Macomb County suburbs.
  Fact 0.6: The film was filmed mostly on location.

Retrieved items:
<item 1>
8 Mile (film)
8 Mile ( film ) - wikipedia 8 Mile ( film ) Jump to : navigation , search 8 Mile Theatrical release poster Directed by Curtis Hanson Produced by Curtis Hanson
Brian Grazer Jimmy Iovine Written by Scott Silver Starring Eminem Kim Basinger Brittany Murphy Mekhi Phifer Music by Eminem Cinematography Rodrigo Prieto Edited
by Jay Rabinowitz Production company Imagine Entertainment Distributed by Universal Pictures Release date November 8 , 2002 ( 2002 - 11 - 08 ) Running time 110
minutes Country United States Language English Budget $41 million Box office $242.9 million 8 Mile is a 2002 American hip hop drama film written by Scott Silver
, directed by Curtis Hanson , and starring Eminem , Mekhi Phifer , Brittany Murphy , Michael Shannon , and Kim Basinger . The film is based loosely on Eminem 's
actual upbringing , and follows white rapper B - Rabbit ( Eminem ) and his attempt to launch a career in a genre dominated by African - Americans .
</item 1>

<item 2>
8 Mile (film)
The title is derived from 8 Mile Road , the highway between the predominantly black city of Detroit and Wayne County and the predominantly White Oakland County
and Macomb County suburbs . It was filmed mostly on location . 8 Mile was a critical and commercial success . Eminem won the Academy Award for Best Original
Song for `` Lose Yourself . '' Contents ( hide ) 1 Plot 2 Cast 3 Music 4 Reception 4.1 Box office 4.2 Critical reception 4.3 Top lists 4.4 Awards and honors 5
References 6 External links Plot ( edit ) In 1995 in Detroit , Jimmy `` B - Rabbit '' Smith Jr. is a blue - collar worker from a poor family . He has moved back
north of 8 Mile Road to the run - down trailer home in Warren , Michigan of his alcoholic mother Stephanie , his little sister Lily , and Stephanie 's abusive
live - in boyfriend Greg . Although encouraged by his friends , Jimmy worries about his potential as a rapper . He falters during a rap battle one night at a
local venue , the Shelter , and he leaves the stage humiliated . Jimmy works at a car factory .
</item 2>

<item 3>
8 Mile (film)
Ronnie Specter , Matiki Anoff MTV Movie Awards Best Movie Nominated 8 Mile Best Male Performance Won Eminem Breakthrough Male Performance Won Eminem Online Film
Critics Society Awards Best Breakthrough Performance Nominated Eminem Phoenix Film Critics Society Awards Best Original Song -- Lose Yourself Nominated Eminem
Satellite Awards Best Original Song -- Lose Yourself Nominated Eminem Teen Choice Awards Choice Movie : Drama Nominated 8 Mile Choice Movie Actor : Drama Won
Eminem Choice Movie : Male Breakout Star Won Eminem Choice Crossover Artist Nominated Eminem Choice Movie : Liplock Nominated Eminem and Brittany Murphy World
Soundtrack Awards Best Original Song Written for a Film -- Lose Yourself Nominated Eminem ASCAP Awards Most Performed Song from a Motion Picture -- Lose
Yourself Won Eminem Grammy Award Grammy Award for Record of the Year -- Lose Yourself Nominated Eminem Grammy Award for Song of the Year -- Lose Yourself
Nominated Jeff Bass , Eminem & Luis Resto Grammy Award for Best Rap Song -- Lose Yourself Won Jeff Bass , Eminem & Luis Resto Grammy Award for Best Male Rap
Solo Performance -- Lose Yourself Won Eminem Grammy Award for Best Song Written for Visual Media -- Lose Yourself Nominated Jeff Bass ,
</item 3>

<item 4>
8 Mile (film)
Jimmy works at a car factory . When he asks for extra shifts , his supervisor laughs and dismisses his request on account of his habitual lateness . Jimmy
befriends a woman named Alex . Over time , Jimmy begins to take more responsibility for the direction of his life . When he exhibits an improved attitude and
performance at work , his supervisor grants him the extra shifts he requested . Stephanie receives a notice evicting her and the family as she can not pay rent
. Despite Stephanie 's best attempts to keep the eviction notice a secret , Greg finds out about it . A fight ensues between him and Jimmy , and Greg leaves
Stephanie for good . Jimmy 's friendship with Wink , a radio DJ with ties to a record label promoter , becomes strained after he discovers that Wink does
promotional work for Jimmy 's rivals , a rap group known as the `` Leaders of the Free World . '' At one point , Jimmy and his friends get into a violent brawl
with the Leaders , which is disrupted when Jimmy 's friend Cheddar Bob pulls out a gun and accidentally shoots himself in the leg ; he survives but is confined
to crutches .
</item 4>

<item 5>
8 Mile (film)
He praised Hanson 's directing and the performances , and compared the final battle with Papa Doc to the fight between Rocky Balboa and Apollo Creed in Rocky .
Top lists ( edit ) 8 Mile has been named in various year - end and all - time top lists : 2nd -- Billboard ( Erika Ramirez ) : Top 10 Best Hip - Hop Movies Ever
7th -- The New York Observer ( Andrew Sarris ) : The 10 Best English - Language Films of 2002 9th -- Time ( Richard Schickel ) : Top 10 Movies of 2002 10th --
Rolling Stone ( Peter Travers ) : The Best Movies of 2002 N / A -- The Daily Californian : Best Films of 2002 Awards and honors ( edit ) In 2003 , Eminem won
the Academy Award for Best Original Song at the 75th Academy Awards , for his single `` Lose Yourself '' from the soundtrack of 8 Mile . Thus he became the
first hip hop artist ever to win an Academy Award . He was not present at the ceremony , but musician Luis Resto accepted the award .
</item 5>

<item 6>
8 Mile (film)
Eminem features on five tracks from the album . It was released under the Shady / Interscope label and spawned Eminem 's first number 1 US single `` Lose
Yourself '' . The album debuted at number one on the U.S. Billboard 200 Albums Chart that year with over 702,000 copies sold and 507,000 sold in the second week
also finishing the year as the fifth best - selling album of 2002 with US sales of 3.2 million , despite only two months of release . It also reached # 1 on the
UK Compilations Chart and the Australian ARIAnet Albums Chart . It also spawned a follow up soundtrack , More Music from 8 Mile , consisting of songs that
appear in 8 Mile that were current singles during the film 's time setting of 1995 . The album was also made in a clean edition removing most of the strong
profanity and violent content . Reception ( edit ) Box Office ( edit ) 8 Mile opened at # 1 with $51,240,555 in its opening weekend , the then second highest
opening for an R - rated movie in the U.S.
</item 6>

<item 7>
8 Mile (film)
Eminem & Luis Resto The film is recognized by American Film Institute in these lists : 2004 : AFI 's 100 Years ... 100 Songs : `` Lose Yourself '' -- # 93 2006
: AFI 's 100 Years ... 100 Cheers -- Nominated References ( edit ) Jump up ^ `` 8 Mile '' . British Board of Film Classification . Retrieved 2012 - 01 - 29 . ^
Jump up to : 8 Mile at Box Office Mojo Jump up ^ `` Eminem . '' Encyclopedia of Popular Music , 4th ed . Ed . Colin Larkin . Oxford Music Online . Oxford
University Press . Web . 27 Sep. 2016 . Jump up ^ `` Eminem movie tops at US box office '' . RTÉ. 2002 - 11 - 12 . Retrieved 2011 - 10 - 06 . Jump up ^ Hettrick
, Scott ( 2003 - 03 - 24 ) . `` ' 8 Mile ' DVD 's going far '' . Variety .
</item 7>

<item 8>
8 Mile (film)
Mekhi Phifer as Future , Jimmy 's closest friend and the rap battle host Brittany Murphy as Alex Latourno , Jimmy 's love interest Kim Basinger as Stephanie ,
Jimmy and Lily 's mother Anthony Mackie as Papa Doc Michael Shannon as Greg Buehl , Stephanie 's boyfriend Eugene Byrd as Wink Evan Jones as Cheddar Bob , Jimmy
's friend Omar Benson Miller as Sol George , Jimmy 's friend and Iz 's brother De'Angelo Wilson as DJ Iz , Jimmy 's friend and Sol 's brother Taryn Manning as
Janeane , Jimmy 's ex-girlfriend Proof as Lil ' Tic Xzibit as Mike ( Male Lunch Truck Rapper ) Craig Chandler as Paul Obie Trice as Male parking lot rapper
Chloe Greenfield as Lily , Jimmy 's sister John Singleton as Bouncer Miz Korona as Vanessa ( Female Lunch Truck Rapper ) Brandon T. Jackson ( uncredited ) as a
Chin Tiki club - foe Music ( edit ) Main article : Music from and Inspired by the Motion Picture 8 Mile Music from and Inspired by the Motion Picture 8 Mile is
the soundtrack to 8 Mile .
</item 8>

<item 9>
8 Mile (film)
In all rounds of the rap battle , Jimmy has to compete against one member of `` the Leaders . '' After winning the first two rounds , he has to challenge Papa
Doc . He wins by exposing his opponent 's weaknesses as well as his own . With nothing to say in rebuttal , Papa Doc hands the microphone back to Future ,
embarrassed . After being congratulated by Alex and his friends , Jimmy is offered a position by Future , hosting battles at The Shelter . Jimmy declines ,
saying he has to get back to work and to find success his own way . Cast ( edit ) Eminem as Jimmy `` B - Rabbit '' Smith , Jr .
</item 9>

<item 10>
8 Mile (film)
He said that Eminem , as an actor , is `` convincing without being too electric '' and `` survives the X-ray truth - telling of the movie camera '' In the At
the Movies with Ebert and Roeper review , both Ebert and Richard Roeper gave the film a thumbs up ; Roeper said that Eminem has a `` winning screen presence ''
and `` raw magic '' to him . He was happy with Rabbit 's `` tender side '' presented through his relationship with the `` adorable '' Greenfield as his sister ,
but felt that Basinger was `` really miscast '' . Roeper said : `` 8 Mile probably wo n't win converts to rap , but it should thrill Eminem fans . '' Peter
Travers gave the film 3.5 out of 4 stars . He said that 8 Mile `` is a real movie , not a fast - buck package to exploit the fan base of a rap nonentity '' that
`` qualifies as a cinematic event by tapping into the roots of Eminem and the fury and feeling that inform his rap . ''
</item 10>
=== END ITEM id=b424633de01528ad ===

=== ITEM id=5fcaff4bbb0c1197 ===
Gold sentences and facts:
Sentence 0: In healthy adults , there are two normal heart sounds , often described as a lub and a dub ( or dup ) , that occur in sequence with each heartbeat . These are the first heart sound ( S ) and second heart sound ( S ) , produced by the closing of the atrioventricular valves and semilunar valves , respectively . In addition to these normal sounds , a variety of other sounds may be present including heart murmurs , adventitious sounds , and gallop rhythms S and S .
  Fact 0.0: In healthy adults there are two normal heart sounds.
  Fact 0.1: The two normal heart sounds are often described as a lub and a dub (or dup).
  Fact 0.2: The two normal heart sounds occur in sequence with each heartbeat.
  Fact 0.3: The first heart sound (S) is produced by the closing of the atrioventricular valves.
  Fact 0.4: The second heart sound (S) is produced by the closing of the semilunar valves.
  Fact 0.5: The closing of the atrioventricular valves and the semilunar valves produces the first and second heart sounds, respectively.
  Fact 0.6: In addition to the normal sounds, a variety of other sounds may be present.
  Fact 0.7: The other sounds that may be present include heart murmurs, adventitious sounds, and gallop rhythms labeled S and S.

Retrieved items:
<item 1>
Heart sounds
Aortic valve ( to aorta ) right second intercostal space upper right sternal border Pulmonary valve ( to pulmonary trunk ) left second intercostal space upper
left sternal border Erb 's point Left third intercostal space left sternal border Tricuspid valve ( to right ventricle ) left fourth , fifth intercostal spaces
lower left sternal border Mitral valve ( to left ventricle ) left fifth intercostal space left midclavicular line Recording heart sounds ( edit ) Using
electronic stethoscopes , it is possible to record heart sounds via direct output to an external recording device , such as a laptop or MP3 recorder . The same
connection can be used to listen to the previously recorded auscultation through the stethoscope headphones , allowing for more detailed study of murmurs and
other heart sounds , for general research as well as evaluation of a particular patient 's condition .
</item 1>

<item 2>
Heart sounds
See also ( edit ) Pulse Physical examination Precordial examination Benign pediatric heart murmur Iambic pentameter , a metre in poetry which follows a similar
rhythm to the human heartbeat Pulsatile tinnitus - hearing a heartbeat sound in one or both ears Notes and references ( edit ) Jump up ^
https://depts.washington.edu/physdx/heart/tech2.html Jump up ^ https://depts.washington.edu/physdx/heart/tech2.html Jump up ^ `` thrill '' . Medline Plus
Medical Dictionary . Jump up ^ Bernstein , Lenny ( 2016 - 01 - 02 ) , `` Heart doctors are listening for clues to the future of their stethoscopes '' ,
Washington Post . External links ( edit ) University of Michigan Heart Sound and Murmur Library . Auscultation Assistant . UCLA Heart Sounds - Heart Murmurs .
practicalclinicalskills.com Physiology of the cardiovascular system Heart Cardiac output Cardiac cycle Cardiac output Heart rate Stroke volume Stroke volume End
- diastolic volume End - systolic volume Afterload Preload Frank -- Starling law Cardiac function curve Venous return curve Wiggers diagram Pressure volume
diagram Ultrasound Fractional
</item 2>

<item 3>
Heart sounds
Main article : Third heart sound Rarely , there may be a third heart sound also called a protodiastolic gallop , ventricular gallop , or informally the ``
Kentucky '' gallop as an onomatopoeic reference to the rhythm and stress of S1 followed by S2 and S3 together ( S1 = Ken ; S2 = tuck ; S3 = y ) . `` lub - dub -
ta '' or `` slosh - ing - in '' If new , indicates heart failure or volume overload . It occurs at the beginning of diastole after S2 and is lower in pitch than
S1 or S2 as it is not of valvular origin . The third heart sound is benign in youth , some trained athletes , and sometimes in pregnancy but if it re-emerges
later in life it may signal cardiac problems , such as a failing left ventricle as in dilated congestive heart failure ( CHF ) . S3 is thought to be caused by
the oscillation of blood back and forth between the walls of the ventricles initiated by blood rushing in from the atria .
</item 3>

<item 4>
Heart sounds
The reason the third heart sound does not occur until the middle third of diastole is probably that during the early part of diastole , the ventricles are not
filled sufficiently to create enough tension for reverberation . It may also be a result of tensing of the chordae tendineae during rapid filling and expansion
of the ventricle . In other words , an S3 heart sound indicates increased volume of blood within the ventricle . An S3 heart sound is best heard with the bell -
side of the stethoscope ( used for lower frequency sounds ) . A left - sided S3 is best heard in the left lateral decubitus position and at the apex of the
heart , which is normally located in the 5th left intercostal space at the midclavicular line . A right - sided S3 is best heard at the lower - left sternal
border . The way to distinguish between a left and right - sided S3 is to observe whether it increases in intensity with inhalation or exhalation . A right -
sided S3 will increase on inhalation , while a left - sided S3 will increase on exhalation .
</item 4>

<item 5>
Heart sounds
Murmurs may be physiological ( benign ) or pathological ( abnormal ) . Abnormal murmurs can be caused by stenosis restricting the opening of a heart valve ,
resulting in turbulence as blood flows through it . Abnormal murmurs may also occur with valvular insufficiency ( regurgitation ) , which allows backflow of
blood when the incompetent valve closes with only partial effectiveness . Different murmurs are audible in different parts of the cardiac cycle , depending on
the cause of the murmur . Contents ( hide ) 1 Primary heart sounds 2 Extra heart sounds 3 Murmurs 3.1 Effects of breathing 3.2 Interventions that change murmurs
4 Other abnormal sounds 5 Surface anatomy 6 Recording heart sounds 7 See also 8 Notes and references 9 External links Primary heart sounds ( edit ) Normal heart
sounds are associated with heart valves closing : S ( edit ) The first heart sound , or S , forms the `` lub '' of `` lub - dub '' and is composed of components
M ( mitral valve closure ) and T ( tricuspid valve closure ) . Normally M precedes T slightly .
</item 5>

<item 6>
Heart sounds
Heart sounds - wikipedia Heart sounds Jump to : navigation , search Normal heart sounds Normal heart sounds as heard with a stethoscope Problems playing this
file ? See media help . 72 BPM Heart sound Front of thorax , showing surface relations of bones , lungs ( purple ) , pleura ( blue ) , and heart ( red outline )
. The locations of best auscultation for each heart valve are labeled with `` M '' , `` T '' , `` A '' , and `` P '' . First heart sound : caused by
atrioventricular valves - Mitral ( M ) and Tricuspid ( T ) . Second heart sound caused by semilunar valves -- Aortic ( A ) and Pulmonary / Pulmonic ( P ) .
Diagram showing relations of opened heart to front of thoracic wall . Ant . Anterior segment of tricuspid valve . A O. Aorta . A.P. Anterior papillary muscle .
In . Innominate artery . L.C.C. Left common carotid artery . L.S. Left subclavian artery . L.V. Left ventricle . P.A. Pulmonary artery .
</item 6>

<item 7>
Heart sounds
L.V. Left ventricle . P.A. Pulmonary artery . R.A. Right atrium . R.V. Right ventricle . V.S. Ventricular septum . Heart sounds are the noises generated by the
beating heart and the resultant flow of blood through it . Specifically , the sounds reflect the turbulence created when the heart valves snap shut . In cardiac
auscultation , an examiner may use a stethoscope to listen for these unique and distinct sounds that provide important auditory data regarding the condition of
the heart . In healthy adults , there are two normal heart sounds , often described as a lub and a dub ( or dup ) , that occur in sequence with each heartbeat .
These are the first heart sound ( S ) and second heart sound ( S ) , produced by the closing of the atrioventricular valves and semilunar valves , respectively
. In addition to these normal sounds , a variety of other sounds may be present including heart murmurs , adventitious sounds , and gallop rhythms S and S .
Heart murmurs are generated by turbulent flow of blood , which may occur inside or outside the heart .
</item 7>

<item 8>
Heart sounds
S3 can be a normal finding in young patients but is generally pathologic over the age of 40 . The most common cause of pathologic S3 is congestive heart failure
. Main article : Fourth heart sound S4 when audible in an adult is called a presystolic gallop or atrial gallop . This gallop is produced by the sound of blood
being forced into a stiff or hypertrophic ventricle . `` ta - lub - dub '' or `` a-stiff - wall '' It is a sign of a pathologic state , usually a failing or
hypertrophic left ventricle , as in systemic hypertension , severe valvular aortic stenosis , and hypertrophic cardiomyopathy . The sound occurs just after
atrial contraction at the end of diastole and immediately before S1 , producing a rhythm sometimes referred to as the `` Tennessee '' gallop where S4 represents
the `` Ten - '' syllable . It is best heard at the cardiac apex with the patient in the left lateral decubitus position and holding his breath . The combined
presence of S3 and S4 is a quadruple gallop , also known as the `` Hello - Goodbye '' gallop .
</item 8>

<item 9>
Heart sounds
This is a characteristic scratching , creaking , high - pitched sound emanating from the rubbing of both layers of inflamed pericardium . It is the loudest in
systole , but can often be heard at the beginning and at the end of diastole . It is very dependent on body position and breathing , and changes from hour to
hour . Surface anatomy ( edit ) The aortic area , pulmonic area , tricuspid area and mitral area are areas on the surface of the chest where the heart is
auscultated . Heart sounds result from reverberation within the blood associated with the sudden block of flow reversal by the valves closing . Because of this
, auscultation to determine function of a valve is usually not performed at the position of the valve , but at the position to where the sound waves reverberate
.
</item 9>

<item 10>
Heart sounds
The sudden tensing of the chordae tendineae and the squeezing of the ventricles against closed semilunar valves , sends blood rushing back toward the atria ,
and the parachute - like valves catch the rush of blood in their leaflets causing the valve to snap shut . The S1 sound results from reverberation within the
blood associated with the sudden block of flow reversal by the valves . The delay of T1 even more than normally causes the split S1 which is heard in a right
bundle branch blockage . The second heart sound , or S , forms the `` dub '' of `` lub - dub '' and is composed of components A ( aortic valve closure ) and P (
pulmonary valve closure ) . Normally A precedes P especially during inspiration where a split of S can be heard . It is caused by the closure of the semilunar
valves ( the aortic valve and pulmonary valve ) at the end of ventricular systole and the beginning of ventricular diastole . As the left ventricle empties ,
its pressure falls below the pressure in the aorta .
</item 10>
=== END ITEM id=5fcaff4bbb0c1197 ===
